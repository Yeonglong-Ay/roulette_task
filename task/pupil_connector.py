# pupil_connector.py
"""
Connects to Pupil Capture via ZeroMQ and sends remote annotations
time-locked to task events.

Architecture
------------
Pupil Capture runs on the same machine and listens on port 50020.
This module uses two ZeroMQ sockets:

    REQ socket → Pupil Remote (port 50020)
        Used to: get the PUB port, get current Pupil time, start/stop
        recording if needed. One request at a time (REQ-REP pattern).

    PUB socket → IPC Backbone (port returned by Pupil Remote)
        Used to: publish annotation messages. Non-blocking.

Annotation format (Pupil Labs remote annotation spec)
------------------------------------------------------
    {
        'topic'    : 'annotation',
        'label'    : '<event label>',
        'timestamp': <float, in Pupil timebase>,
        'duration' : 0.0,
    }

Time synchronisation
--------------------
At startup, the connector queries Pupil Capture for its current clock
('t' command) and records the local time at that moment. The offset
    pupil_offset = pupil_time - local_time
is then used to convert any subsequent local timestamp to Pupil time:
    pupil_ts = time.time() + pupil_offset

This means annotations are always in Pupil's timebase, regardless of
clock drift between startup and the event.

Fallback
--------
If Pupil Capture is not running or zmq/msgpack are not installed,
all methods become silent no-ops. The task continues normally.

Dependencies
------------
    pip install pyzmq msgpack
"""

from typing import Optional
import threading
import time


class PupilConnector:
    """
    Interface to Pupil Capture for sending remote annotations.

    Usage
    -----
        pupil = PupilConnector(enabled=True)
        pupil.connect()            # call once after TriggerSender init

        # At each task event:
        pupil.annotate('BET_ONSET', extra={'trial': 1, 'streak': 'L3'})

        pupil.close()              # at session end
    """

    PUPIL_PORT    = 50020
    CONNECT_TIMEOUT_S = 2.0

    def __init__(self, enabled: bool = True):
        self.enabled       = enabled
        self._connected    = False
        self._pupil_remote = None   # REQ socket
        self._publisher    = None   # PUB socket
        self._ctx          = None
        self._time_offset  = 0.0    # pupil_time - local_time
        self._log: list[dict] = []

        if not enabled:
            print("[PupilConnector] Disabled.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Connect to Pupil Capture on localhost:50020.
        Returns True if successful, False if Pupil Capture is not running.
        """
        if not self.enabled:
            return False

        try:
            import zmq
            self._ctx = zmq.Context()

            # REQ socket for Pupil Remote commands
            self._pupil_remote = self._ctx.socket(zmq.REQ)
            self._pupil_remote.setsockopt(zmq.RCVTIMEO, int(self.CONNECT_TIMEOUT_S * 1000))
            self._pupil_remote.connect(f'tcp://localhost:{self.PUPIL_PORT}')

            # Verify connection with a version request
            self._pupil_remote.send_string('v')
            version = self._pupil_remote.recv_string()
            print(f"[PupilConnector] Connected to Pupil Capture {version}")

            # Get PUB port for the IPC Backbone
            self._pupil_remote.send_string('PUB_PORT')
            pub_port = self._pupil_remote.recv_string()

            # PUB socket for annotation messages
            self._publisher = self._ctx.socket(zmq.PUB)
            self._publisher.connect(f'tcp://localhost:{pub_port}')

            # Allow publisher to establish (ZMQ async connect)
            time.sleep(0.1)

            # Compute time offset (Pupil time vs local time)
            self._sync_time()

            self._connected = True
            print(
                f"[PupilConnector] Ready.  "
                f"Time offset = {self._time_offset:+.4f} s  "
                f"(pub port = {pub_port})"
            )
            return True

        except ImportError:
            print("[PupilConnector] pyzmq or msgpack not installed — "
                  "eye tracking annotations disabled.")
        except Exception as e:
            print(f"[PupilConnector] Could not connect to Pupil Capture: {e}")
            print("  → Ensure Pupil Capture is running and the Network API "
                  "plugin is enabled.")
        self._cleanup()
        return False

    def annotate(self, label: str, extra: Optional[dict] = None) -> None:
        """
        Send a remote annotation to Pupil Capture.

        The annotation timestamp is computed using the local clock plus
        the Pupil time offset measured at connect().

        Args:
            label : Event label (e.g. 'BET_ONSET', 'FEEDBACK_ONSET').
            extra : Optional dict of extra fields added to the annotation
                    (exported in Pupil Player's annotation CSV).
        """
        if not self.enabled or not self._connected:
            return

        pupil_ts = time.time() + self._time_offset

        self._log.append({
            'local_time' : time.time(),
            'pupil_time' : pupil_ts,
            'label'      : label,
        })

        # Build annotation dict
        annotation = {
            'topic'    : 'annotation',
            'label'    : label,
            'timestamp': pupil_ts,
            'duration' : 0.0,
        }
        if extra:
            annotation.update(extra)

        # Send non-blocking in daemon thread to avoid delaying the task
        threading.Thread(
            target=self._publish_annotation,
            args=(annotation,),
            daemon=True,
        ).start()

    def save_log(self, path) -> None:
        """Write annotation log to TSV."""
        if not self._log:
            return
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('local_time\tpupil_time\tlabel\n')
            for row in self._log:
                f.write(
                    f"{row['local_time']:.6f}\t"
                    f"{row['pupil_time']:.6f}\t"
                    f"{row['label']}\n"
                )
        print(f"[PupilConnector] Annotation log saved ({len(self._log)} events) → {path}")

    def close(self) -> None:
        """Release ZMQ sockets."""
        self._cleanup()
        print("[PupilConnector] Disconnected.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_time(self) -> None:
        """Query Pupil Capture clock and compute offset vs local clock."""
        try:
            t_before = time.time()
            self._pupil_remote.send_string('t')
            pupil_ts  = float(self._pupil_remote.recv_string())
            t_after   = time.time()
            # Use midpoint of the round-trip as the local reference
            local_mid = (t_before + t_after) / 2
            self._time_offset = pupil_ts - local_mid
        except Exception as e:
            print(f"[PupilConnector] Time sync failed: {e}")
            self._time_offset = 0.0

    def _publish_annotation(self, annotation: dict) -> None:
        """Publish one annotation to the IPC Backbone (runs in thread)."""
        if self._publisher is None:
            return
        try:
            import msgpack
            topic   = 'annotation'
            payload = msgpack.dumps(annotation, use_bin_type=True)
            self._publisher.send_string(topic, flags=1)   # zmq.SNDMORE = 1
            self._publisher.send(payload)
        except Exception as e:
            print(f"[PupilConnector] Annotation send error: {e}")

    def _cleanup(self) -> None:
        """Close sockets safely."""
        for sock in [self._pupil_remote, self._publisher]:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
        self._pupil_remote = None
        self._publisher    = None
        self._ctx          = None
        self._connected    = False

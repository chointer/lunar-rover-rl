import sys
import tty
import termios
import threading
import time

KEY_UP    = 265
KEY_DOWN  = 264
KEY_LEFT  = 263
KEY_RIGHT = 262

# 터미널 ESC 시퀀스 → 키코드 매핑 (화살표: \x1b[A/B/C/D)
_SEQ_MAP = {b'[A': KEY_UP, b'[B': KEY_DOWN, b'[D': KEY_LEFT, b'[C': KEY_RIGHT}


class KeyboardInput:
    """터미널 stdin에서 화살표 키 입력을 읽는 클래스.

    WSL에서 GLFW REPEAT 이벤트가 오지 않는 문제를 우회하기 위해
    OS 레벨 키 반복을 활용하는 터미널 stdin 방식 사용.
    사용 시 터미널 창에 포커스를 유지해야 함.
    """

    def __init__(self):
        self._key_times = {}
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.buffer.read(1)
                if ch == b'\x03':   # Ctrl+C → 종료
                    break
                if ch == b'\x1b':
                    code = _SEQ_MAP.get(sys.stdin.buffer.read(2))
                    if code:
                        self._key_times[code] = time.perf_counter()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def is_held(self, keycode, timeout=0.6):
        """키가 눌린 상태인지 반환.

        터미널 키 반복 주기 ~30ms, 초기 딜레이 ~500ms → timeout 0.6s로 커버.
        """
        return time.perf_counter() - self._key_times.get(keycode, 0) < timeout

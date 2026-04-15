#!/usr/bin/env python3
"""
ddebug 客户端 — 连接 ddebug() 断点。

支持：方向键编辑、上下键历史、Ctrl+C 取消、Ctrl+D 退出、跨会话记忆。

用法：
    python ddebug_connect.py          # 默认端口 12345
    python ddebug_connect.py 12346    # 指定端口
"""

import socket
import sys
import os
import threading
import time
import queue

HISTORY_FILE = os.path.expanduser("~/.ddebug_history")

try:
    import readline
    readline.set_history_length(1000)
    if os.path.exists(HISTORY_FILE):
        readline.read_history_file(HISTORY_FILE)
except ImportError:
    pass


def _save_history():
    try:
        readline.write_history_file(HISTORY_FILE)
    except Exception:
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError:
        print(f"Cannot connect to {host}:{port} — is ddebug() waiting?")
        sys.exit(1)

    print(f"Connected to ddebug at {host}:{port}\n")

    # ---- 接收线程只负责把数据放进队列，不做任何打印 ----
    data_queue = queue.Queue()
    alive = True

    def receiver():
        nonlocal alive
        while alive:
            try:
                data = sock.recv(8192)
                if not data:
                    alive = False
                    data_queue.put(None)  # 哨兵值
                    return
                data_queue.put(data.decode("utf-8", errors="replace"))
            except OSError:
                alive = False
                data_queue.put(None)
                return

    t = threading.Thread(target=receiver, daemon=True)
    t.start()

    def read_until_prompt():
        """
        从队列中读取数据，累积直到发现末尾的 prompt。
        返回 (output, prompt)：output 是要打印的内容，prompt 是 ">>> " 或 "... "。
        如果连接关闭返回 (remaining_text, None)。
        """
        buf = ""
        while True:
            try:
                chunk = data_queue.get(timeout=60)
            except queue.Empty:
                # 超时，打印已有内容
                return buf, ">>> "

            if chunk is None:
                return buf, None  # 连接关闭

            buf += chunk

            # 检查是否以 prompt 结尾
            for prompt_str in (">>> ", "... "):
                if buf.endswith(prompt_str):
                    output = buf[:-len(prompt_str)]
                    return output, prompt_str

    # ---- 主循环：所有打印都在主线程，不存在竞争 ----

    # 先读 banner + 第一个 prompt
    output, prompt = read_until_prompt()
    if output:
        sys.stdout.write(output)
        sys.stdout.flush()

    if prompt is None:
        print("[Connection closed]")
        sock.close()
        return

    while alive:
        try:
            line = input(prompt)
            if not alive:
                break
            sock.sendall((line + "\n").encode("utf-8"))

            # 读取服务端的完整响应（输出 + 下一个 prompt）
            output, prompt = read_until_prompt()
            if output:
                sys.stdout.write(output)
                sys.stdout.flush()
            if prompt is None:
                print("[Connection closed by server]")
                break

        except KeyboardInterrupt:
            sys.stdout.write("\nKeyboardInterrupt\n")
            sys.stdout.flush()
            try:
                sock.sendall(b"\n")
            except OSError:
                break
            # 读取服务端对空行的响应
            output, prompt = read_until_prompt()
            if output:
                sys.stdout.write(output)
                sys.stdout.flush()
            if prompt is None:
                break

        except EOFError:
            print("\n[Exiting]")
            break

    sock.close()
    _save_history()
    print("Disconnected.")


if __name__ == "__main__":
    main()
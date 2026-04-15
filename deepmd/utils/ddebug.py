"""
Distributed Debug Utility — ddebug()

Usage:
    from ddebug import ddebug

    # 在多卡训练代码的任意位置插入：
    ddebug()

    # 程序会打印连接指令，在另一个终端执行：
    python ddebug_connect.py

    # 进入交互后，可用的变量和函数：
    #   loc["x"]              → rank 0 本地的 x (原始对象，在 GPU 上)
    #   r[1]["x"]             → rank 1 上的 x (CPU 副本)
    #   diff("x")             → 所有 rank 的 x 与 rank 0 的最大差异
    #   diff("grad_x", 0, 2)  → rank 0 vs rank 2 的 grad_x 差异
    #   show("loss")          → 打印所有 rank 上 loss 的值
    #   shapes("y")           → 打印所有 rank 上 y 的 shape
"""

import sys
import inspect
import socket
import code
import torch
import torch.distributed as dist
import pickle


def _gather_object(obj, world_size):
    out = [None] * world_size
    dist.all_gather_object(out, obj)
    return out


def _serialize_locals(local_vars: dict) -> dict:
    snapshot = {}
    for name, val in local_vars.items():
        if name.startswith("_"):
            continue
        try:
            if isinstance(val, torch.Tensor):
                snapshot[name] = val.detach().cpu().clone()
            elif isinstance(val, (int, float, str, bool, list, tuple, dict, type(None))):
                snapshot[name] = val
            elif isinstance(val, torch.nn.Module):
                nparams = sum(p.numel() for p in val.parameters())
                snapshot[name] = f"<Module {val.__class__.__name__}, {nparams} params>"
            else:
                try:
                    pickle.dumps(val)
                    snapshot[name] = val
                except Exception:
                    snapshot[name] = repr(val)
        except Exception:
            snapshot[name] = f"<unserializable: {type(val).__name__}>"
    return snapshot


class _RankAccessor:
    """r[i]["var_name"] 获取 rank i 上的变量"""

    def __init__(self, all_snapshots):
        self._data = all_snapshots

    def __getitem__(self, rank):
        if rank < 0 or rank >= len(self._data):
            raise IndexError(f"rank {rank} out of range [0, {len(self._data)})")
        return self._data[rank]

    def __repr__(self):
        lines = []
        for i, snap in enumerate(self._data):
            tensor_keys = [k for k, v in snap.items() if isinstance(v, torch.Tensor)]
            other_keys = [k for k, v in snap.items() if not isinstance(v, torch.Tensor)]
            lines.append(f"  rank {i}: tensors={tensor_keys}, others={other_keys}")
        return "RankAccessor(\n" + "\n".join(lines) + "\n)"


class _SocketIO:
    """把 socket 包装成 file-like 对象"""

    def __init__(self, conn):
        self.conn = conn
        self._buffer = ""

    def write(self, data):
        try:
            self.conn.sendall(data.encode("utf-8"))
        except (BrokenPipeError, OSError):
            pass

    def readline(self, prompt=""):
        if prompt:
            self.write(prompt)
        while "\n" not in self._buffer:
            chunk = self.conn.recv(4096)
            if not chunk:
                raise EOFError
            text = chunk.decode("utf-8")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            self._buffer += text
        line, self._buffer = self._buffer.split("\n", 1)
        return line

    def flush(self):
        pass


def ddebug(target_rank=0, port=12345):
    """
    分布式 debug 断点。插入 ddebug() 即可。

    所有 rank 参与 gather，但只有 target_rank 打开调试。
    在另一个终端运行 `python ddebug_connect.py` 连接。

    Args:
        target_rank: 在哪个 rank 上调试 (default 0)
        port: TCP 端口 (default 12345)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    frame = inspect.currentframe().f_back
    caller_locals = dict(frame.f_locals)

    snapshot = _serialize_locals(caller_locals)
    all_snapshots = _gather_object(snapshot, world_size)

    if rank == target_rank:
        r = _RankAccessor(all_snapshots)
        loc = caller_locals

        def show(name):
            """打印所有 rank 上某变量的值"""
            for i in range(world_size):
                val = all_snapshots[i].get(name, "<not found>")
                if isinstance(val, torch.Tensor) and val.numel() <= 20:
                    print(f"  rank {i}: {val}")
                elif isinstance(val, torch.Tensor):
                    print(f"  rank {i}: shape={val.shape}, "
                          f"mean={val.mean():.6f}, std={val.std():.6f}, "
                          f"min={val.min():.6f}, max={val.max():.6f}")
                else:
                    print(f"  rank {i}: {val}")

        def diff(name, rank_a=None, rank_b=None):
            """对比 tensor: diff("x") 全部 vs rank0, diff("x",1,3) 指定两个"""
            if rank_a is not None and rank_b is not None:
                a = all_snapshots[rank_a].get(name)
                b = all_snapshots[rank_b].get(name)
                if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                    print(f"  '{name}' is not a tensor on one of the ranks")
                    return
                if a.shape != b.shape:
                    print(f"  Shape mismatch: rank {rank_a}={a.shape}, rank {rank_b}={b.shape}")
                    return
                d = (a - b).abs()
                print(f"  rank {rank_a} vs rank {rank_b}: "
                      f"max_diff={d.max():.2e}, mean_diff={d.mean():.2e}")
            else:
                ref = all_snapshots[0].get(name)
                if not isinstance(ref, torch.Tensor):
                    print(f"  '{name}' is not a tensor on rank 0")
                    return
                for i in range(1, world_size):
                    other = all_snapshots[i].get(name)
                    if not isinstance(other, torch.Tensor):
                        print(f"  rank {i}: not a tensor")
                        continue
                    if ref.shape != other.shape:
                        print(f"  rank 0 vs rank {i}: shape mismatch "
                              f"{ref.shape} vs {other.shape}")
                        continue
                    d = (ref - other).abs()
                    print(f"  rank 0 vs rank {i}: "
                          f"max_diff={d.max():.2e}, mean_diff={d.mean():.2e}")

        def shapes(name):
            """打印所有 rank 上某 tensor 的 shape"""
            for i in range(world_size):
                val = all_snapshots[i].get(name, "<not found>")
                if isinstance(val, torch.Tensor):
                    print(f"  rank {i}: {val.shape} (dtype={val.dtype})")
                else:
                    print(f"  rank {i}: not a tensor ({type(val).__name__})")

        # ---- 启动 TCP 调试服务器 ----
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)

        print(f"\n{'*'*60}")
        print(f"  ddebug: rank {rank}/{world_size}, listening on port {port}")
        print(f"  Open another terminal and run:")
        print(f"")
        print(f"      python ddebug_connect.py {port}")
        print(f"")
        print(f"{'*'*60}\n")
        sys.stdout.flush()

        conn, addr = srv.accept()
        print(f"  ddebug: Connected from {addr}\n")

        sock_io = _SocketIO(conn)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = sock_io
        sys.stderr = sock_io

        banner = (
            f"\n{'='*60}\n"
            f"  DISTRIBUTED DEBUG — rank {rank}/{world_size}\n"
            f"  Variables: {sorted(snapshot.keys())}\n"
            f"{'='*60}\n"
            f"  loc['var']       — local var (original, on-device)\n"
            f"  r[i]['var']      — rank i's var (CPU copy)\n"
            f"  show('var')      — print var on all ranks\n"
            f"  diff('var')      — compare across ranks\n"
            f"  diff('var',1,3)  — rank 1 vs rank 3\n"
            f"  shapes('var')    — shapes on all ranks\n"
            f"  Ctrl+C           — cancel current line\n"
            f"  Ctrl+D / exit()  — quit and continue training\n"
            f"{'='*60}\n"
        )

        debug_ns = {
            "loc": loc,
            "r": r,
            "show": show,
            "diff": diff,
            "shapes": shapes,
            "torch": torch,
            "all_snapshots": all_snapshots,
            "rank": rank,
            "world_size": world_size,
        }

        console = code.InteractiveConsole(locals=debug_ns)
        console.raw_input = sock_io.readline
        try:
            console.interact(banner=banner, exitmsg="Bye! Continuing training...")
        except (SystemExit, EOFError):
            pass
        except Exception as e:
            old_stdout.write(f"\n  ddebug ERROR: {type(e).__name__}: {e}\n")
            old_stdout.flush()
            import traceback
            traceback.print_exc(file=old_stdout)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            conn.close()
            srv.close()
            print("  ddebug: Session ended. Continuing...\n")

    dist.barrier()
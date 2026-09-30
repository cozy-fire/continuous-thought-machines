"""Environment-only spawn pool with quiet pipe closure after peer worker failure."""
import multiprocessing as mp
import traceback
from tasks.continual_nav.envs.evaluation import EvaluationPool as BasePool,_command


def _worker(connection,factory):
    env=None
    try:
        while True:
            command,value=connection.recv()
            if command=='close': break
            env,result=_command(env,command,value,factory)
            connection.send((True,result))
    except (EOFError,BrokenPipeError,ConnectionResetError):
        pass
    except BaseException:
        # Another worker may have failed first and caused the parent to close this pipe.
        # Preserve the primary error without raising a second unhandled send exception.
        try: connection.send((False,traceback.format_exc()))
        except (OSError,EOFError): pass
    finally:
        if env is not None: env.close()
        connection.close()


class EvaluationPool(BasePool):
    def __init__(self,size,backend,*,factory):
        if backend not in ('serial','subprocess'): raise ValueError('invalid evaluation backend')
        super().__init__(size,'serial',factory=factory)
        self.backend=backend
        try:
            if backend=='subprocess':
                context=mp.get_context('spawn')
                for _ in range(size):
                    parent,child=context.Pipe()
                    process=context.Process(target=_worker,args=(child,factory),daemon=True)
                    try: process.start()
                    except BaseException:
                        parent.close(); raise
                    finally: child.close()
                    self.connections.append(parent); self.processes.append(process)
        except BaseException:
            self.close(); raise

import time

class StageTimer:
    def __init__(self, ctx, stage_name):
        self.ctx = ctx
        self.stage_name = stage_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.ctx["timing"][self.stage_name] = end - self.start
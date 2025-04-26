# src/es_drl/utils/logger.py
import os

class Logger:
    def __init__(self, log_dir: str, filename: str = "progress.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self.file = open(self.filepath, "w")
        self.header_written = False
        self.keys = []

    def log(self, step: int, data: dict):
        # On first call, write header
        if not self.header_written:
            self.keys = list(data.keys())
            header = "step," + ",".join(self.keys) + "\n"
            self.file.write(header)
            self.header_written = True

        # Write a line: step, then each value in data in order of self.keys
        values = [str(data[k]) for k in self.keys]
        line = str(step) + "," + ",".join(values) + "\n"
        self.file.write(line)
        self.file.flush()

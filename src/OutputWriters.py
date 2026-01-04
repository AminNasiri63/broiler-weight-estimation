# =========================
# Output Writers
# =========================

from typing import Optional
from abc import ABC, abstractmethod
import csv


class OutputWriter(ABC):
    @abstractmethod
    def write(self, record: dict):
        pass

    @abstractmethod
    def close(self):
        pass


class CSVOutputWriter(OutputWriter):
    def __init__(self, csv_path: str):
        self.csv_file = open(csv_path, "w", newline="")
        self.writer: Optional[csv.DictWriter] = None
        self.csv_path = csv_path

    def write(self, record: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=list(record.keys()))
            self.writer.writeheader()
        self.writer.writerow(record)

    def close(self):
        self.csv_file.close()
        print(f"[CSVOutputWriter] Saved results to {self.csv_path}")
from datetime import timedelta
from pathlib import Path

class StatsManager:
	def __init__(self):
		self.time_for_model = {}

	def set_n_images(self, n: int):
		self.n_images = n
	
	def set_time_for_model(self, model_name: str, time: timedelta):
		self.time_for_model[model_name] = time

	def save(self, out_dir: Path):
		if not hasattr(self, 'n_images') or len(self.time_for_model) == 0:
			raise ValueError("Trying to save without proper attributes is probably not what you want. Aborting")

		out_str = self._to_out_str()
		out_file = out_dir / 'stats.txt'
		with out_file.open('w') as f:
			f.write(out_str)

	def _to_out_str(self):
		stats_str = (
			f"{self.n_images} imagens\n"
			f"{'Modelo'.ljust(10)} {'Tempo total (s)'.ljust(20)} Tempo médio por imagem (s)\n"
		)
		for model_name, total_time in self.time_for_model.items():
			average_time = total_time / self.n_images

			stats_str += f"{model_name.ljust(10)} {str(total_time).ljust(20)} {str(average_time)}\n"

		return stats_str
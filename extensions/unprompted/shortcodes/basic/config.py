import json
import os
from types import SimpleNamespace
class Shortcode():
	def __init__(self,Unprompted):
		self.Unprompted = Unprompted
		self.temp_config = self.Unprompted.Config
		self.description = "Updates your settings with the content for the duration of a run."
	def run_block(self, pargs, kwargs, context, content):
		content = self.Unprompted.parse_alt_tags(content,context)

		# Detect inline JSON string
		if (content[0] == "{"):
			json_obj = json.loads(content)
		# Otherwise we're loading a file
		else:
			filepath = self.Unprompted.parse_filepath(content + ".json",context)
			self.Unprompted.log(f"Loading the following config file: {filepath}")
			json_obj = json.load(open(f"{filepath}", "r", encoding="utf8"))

		# Write new settings
		# self.Unprompted.cfg_dict.update(json_obj)
		import lib_unprompted.flatdict as flatdict
		flat_user_cfg = flatdict.FlatDict(json_obj)
		flat_cfg = flatdict.FlatDict(self.Unprompted.cfg_dict)
		# Write differences to flattened dictionary
		flat_cfg.update(flat_user_cfg)
		# Unflatten
		self.Unprompted.cfg_dict = flat_cfg.as_dict()

		self.Unprompted.Config = json.loads(json.dumps(self.Unprompted.cfg_dict), object_hook=lambda d: SimpleNamespace(**d))
		self.Unprompted.log(f"New config: {self.Unprompted.Config}")
		return("")
	
	def cleanup(self):
		# Revert settings
		self.Unprompted.Config = self.temp_config

	def ui(self,gr):
		pass
class Shortcode():
	def __init__(self,Unprompted):
		self.Unprompted = Unprompted
		self.after_content = []
		self.description = "Processes arbitrary text following the main output."

	def run_block(self, pargs, kwargs, context, content):
		index = int(self.Unprompted.parse_advanced(pargs[0])) if len(pargs) > 0 else 0
		self.after_content.insert(index,content)
		return("")
	
	def after(self,p=None,processed=None):
		if self.after_content:
			if processed:
				# Share variable with other shortcodes
				self.Unprompted.after_processed = processed
				# Fix init_images for other functions that may expect it (e.g. txt2mask)
				self.Unprompted.shortcode_user_vars["init_images"] = self.Unprompted.after_processed.images

			# Set up system var support - copy relevant p attributes into shortcode var object
			for att in dir(p):
				if not att.startswith("__"):
					self.Unprompted.shortcode_user_vars[att] = getattr(p,att)

			for idx,content in enumerate(self.after_content):
				self.Unprompted.shortcode_user_vars["after_index"] = idx
				self.Unprompted.process_string(self.Unprompted.parse_alt_tags(content,"after"))

			self.after_content = []
			return(self.Unprompted.after_processed)

	def ui(self,gr):
		gr.Number(label="Order compared to other [after] blocks 🡢 int",value=0,interactive=True)
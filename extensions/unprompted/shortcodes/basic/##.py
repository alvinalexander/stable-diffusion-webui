class Shortcode():
	def __init__(self,Unprompted):
		self.Unprompted = Unprompted
		self.description = "Houses a multiline comment that will not affect the final output."
	def run_block(self, pargs, kwargs, context, content):
		return("")
	def ui(self,gr):
		pass
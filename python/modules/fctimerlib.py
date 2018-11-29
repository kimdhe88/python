from threading import Timer

class functionTimer:
   def __init__(self,t,func):
      self.t=t
      self.func = func
      self.thread = Timer(self.t,self.handle_function)

   def handle_function(self):
      self.func()
      self.thread = Timer(self.t,self.handle_function)
      self.thread.start()

   def start(self):
      self.thread.start()

   def cancel(self):
      self.thread.cancel()

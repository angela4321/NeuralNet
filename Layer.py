import numpy as np
class Layer:

    def initialize_vars(self,w=None,b=None,z=None,a=None,vdw=None,vdb=None,sdw=None,sdb=None,b1=0.9,b2=0.99,learn = 0.005):
        self.w=w
        self.b=b
        self.z=z
        self.a=a


        #for adam optimization
        self.vdw = vdw
        self.vdb = vdb
        self.sdw = sdw
        self.sdb = sdb

        self.b1=b1
        self.b2=b2

        self.learn = learn

    def adam(self, dw, db,iteration):
        self.vdw = (self.b1) * self.vdw + (1 - self.b1) * dw
        vdwc = self.vdw / (1 - pow(self.b1, iteration + 1))

        self.vdb = self.b1 * self.vdb + (1 - self.b1) * db
        vdbc = self.vdb / (1 - pow(self.b1, iteration + 1))

        self.sdw = self.b2 * self.sdw + (1 - self.b2) * np.power(dw, 2)
        sdwc = self.sdw / (1 - pow(self.b2, iteration + 1))

        self.sdb = self.b2 * self.sdb + (1 - self.b2) * np.power(db, 2)
        sdbc = self.sdb / (1 - pow(self.b2, iteration + 1))

        # update weights
        self.w = self.w - self.learn * vdwc / (np.sqrt(sdwc) + 10 ** -8)
        self.b = self.b - self.learn * vdbc / (np.sqrt(sdbc) + 10 ** -8)

    def cost(self):
        print("No cost function")

    def forward_prop(self):
        pass

    def backward_prop(self):
        pass
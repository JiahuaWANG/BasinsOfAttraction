import math


class PotencialSimpleFactory(object):
    
    @staticmethod
    def create_potencial(name, calc_type, *parameters):
        potencial = None
        
        if calc_type == "CPU":
            
            if name == "Potencial1":
                potencial = Potencial1cpu(parameters[0])
            elif name == "Potencial2":
                potencial = Potencial2cpu(parameters[0])
            
        elif calc_type == "GPU":
            if name == "Potencial1":
                potencial = Potencial1gpu(parameters[0])
            elif name == "Potencial2":
                potencial = Potencial2gpu(parameters[0])
            elif name == "Potencial3":
                potencial = Potencial3gpu(parameters[0])
        
        return potencial
        
        
class PotencialCPU(object):
    def __init__(self, gamma):
        self.gamma = gamma
        self.calc_type = "CPU"

class Potencial1cpu(PotencialCPU):
    
    def diff_eq(self, x, v):
        return [v, -x*(x*x - 1) - self.gamma*v]
    
    def determine_minimum(self, x):
        if x > 0:
            return 1
        else:
            return 0
        
        
class Potencial2cpu(PotencialCPU):
    
    def diff_eq(self, x, v):
        return [v, -(4*x*(x*x - 1) + math.cos(x)) - self.gamma*v]
    
    def determine_minimum(self, x):
        if x > 0.25:
            return 1
        else:
            return 0


class PotencialGPU(object):
    def __init__(self):
        self.calc_type = "GPU"

class Potencial1gpu(PotencialGPU):
    def __init__(self, gamma):
        PotencialGPU.__init__(self)
        self.gpu_source = """
            __device__ inline void diff_eq(float t, float &nx, float &nv, float x, float v) {
                nx = v;
                nv = -x*(x*x - 1) - %sf*v;
            }
            
            __device__ int determineMinimum(float x) {            
                if (x > 0) {
                    return 1;
                } else {
                    return 0;
                }
            }
        """ % (gamma)

   
class Potencial2gpu(PotencialGPU):
    def __init__(self, gamma):
        PotencialGPU.__init__(self)
        self.gpu_source = """
            __device__ inline void diff_eq(float &nx, float &nv, float x, float v) {
                nx = v;
                nv = -(4*x*(x*x - 1) + cos(x)) - %sf*v;
            }
            
            __device__ int determineMinimum(float x) {            
                if (x > 0.25) {
                    return 1;
                } else {
                    return 0;
                }
            }
        """ % (gamma)


class Potencial3gpu(PotencialGPU):
    def __init__(self, gamma):
        PotencialGPU.__init__(self)
        self.gpu_source = """
            __device__ inline void diff_eq(float t, float &nx, float &nv, float x, float v) {
                nx = v;
                nv = -x*(x*x - 1) - %sf*v + 0.35*cos(1.4*t);
            }
            
            __device__ int determineMinimum(float x) {            
                if (x > 0) {
                    return 1;
                } else {
                    return 0;
                }
            }
        """ % (gamma)
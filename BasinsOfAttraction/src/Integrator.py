class IntegratorSimpleFactory(object):
    
    @staticmethod
    def createIntegrator(potencial, integrator_method):
        integrator = None
        
        if potencial.calc_type == "CPU":
            
            if integrator_method == "Euler":
                integrator = EulerIntegratorCPU(potencial) 
            elif integrator_method == "RK4":
                integrator = RK4IntegratorCPU(potencial) 
            
        elif potencial.calc_type == "GPU":
            
            if integrator_method == "Euler":
                integrator = EulerIntegratorGPU(potencial) 
            elif integrator_method == "RK4":
                integrator = RK4IntegratorGPU(potencial)

        print "Integrating using",integrator_method,"method on",potencial.calc_type,"device"

        return integrator


class IntegratorCPU(object):
    
    def __init__(self, potencial, delta_time = 0.01):
        self.potencial = potencial
        self.delta_time = delta_time


class EulerIntegratorCPU(IntegratorCPU):
    
    def calculate_step(self, x, v):
        pv = [x, v]
        dt = self.delta_time

        k = self.potencial.diff_eq(pv[0], pv[1])
        
        return [x + k[0]*dt, v + k[1]*dt] 


class RK4IntegratorCPU(IntegratorCPU):
    
    def calculate_step(self, x, v):
        pv = [x, v]
        dt = self.delta_time

        k1 = self.potencial.diff_eq(pv[0], pv[1])
        k2 = self.potencial.diff_eq(pv[0] + 0.25*k1[0]*dt, pv[1] + 0.25*k1[1]*dt)
        k3 = self.potencial.diff_eq(pv[0] + 0.25*k2[0]*dt, pv[1] + 0.25*k2[1]*dt)
        k4 = self.potencial.diff_eq(pv[0] + k3[0]*dt, pv[1] + k3[1]*dt)
        
        x = x + (dt/6.0) * (k1[0] + 2*(k2[0] + k3[0]) + k4[0])
        v = v + (dt/6.0) * (k1[1] + 2*(k2[1] + k3[1]) + k4[1])
        
        return [x, v]

    
class IntegratorGPU(object):
    
    def __init__(self, potencial, delta_time = 0.01):
        self.potencial = potencial
        self.dt_source = """
            __constant__ float dt = %sf;
        """ % (delta_time)   
        
        self.prepare_gpu_source()
        
    def prepare_gpu_source(self):
        pass 
    
class EulerIntegratorGPU(IntegratorGPU):
    
    def prepare_gpu_source(self):
        self.gpu_source = self.dt_source + """
            __device__ inline void calculateStep(float t, float &x, float &v) {
                float nx, nv;
                
                diff_eq(t, nx, nv, x, v);
        
                x = x + nx*dt;
                v = v + nv*dt;
            }
        """ 
    

class RK4IntegratorGPU(IntegratorGPU):
    
    def prepare_gpu_source(self):
        self.gpu_source = self.dt_source + """
            __device__ inline void calculateStep(float t, float &x, float &v) {
                float pv[2], k1[2], k2[2], k3[2], k4[2];
                float nx, nv;

                pv[0] = x;
                pv[1] = v;
                
                diff_eq(t, nx, nv, pv[0], pv[1]);
                k1[0] = nx;
                k1[1] = nv;

                diff_eq(t, nx, nv, pv[0] + 0.25f*k1[0]*dt, pv[1] + 0.25f*k1[1]*dt);
                k2[0] = nx;
                k2[1] = nv;
                
                diff_eq(t, nx, nv, pv[0] + 0.25f*k2[0]*dt, pv[1] + 0.25f*k2[1]*dt);
                k3[0] = nx;
                k3[1] = nv;
                
                diff_eq(t, nx, nv, pv[0] + k3[0]*dt, pv[1] + k3[1]*dt);
                k4[0] = nx;
                k4[1] = nv;
        
                x = x + (dt/6.0f) * (k1[0] + 2*(k2[0] + k3[0]) + k4[0]);
                v = v + (dt/6.0f) * (k1[1] + 2*(k2[1] + k3[1]) + k4[1]);
            }
        """ 
        
        
        
        
        
        
        
        
        

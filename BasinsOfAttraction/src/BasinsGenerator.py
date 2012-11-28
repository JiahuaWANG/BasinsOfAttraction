#@PydevCodeAnalysisIgnore
import numpy
from Potencial import *
import pycuda.driver as cuda
from pycuda.compiler import SourceModule 
from Integrator import IntegratorSimpleFactory


class GeneratorSimpleFactory(object):
    
    @staticmethod
    def createGenerator(potencial, integrator_method):
        generator = None
        integrator = IntegratorSimpleFactory.createIntegrator(potencial, integrator_method)
        
        if potencial.calc_type == "CPU":
            generator = GeneratorCPU(integrator)
            
        elif potencial.calc_type == "GPU":
            generator = GeneratorGPU(integrator)

        return generator
    

class Generator(object):

    def __init__(self, integrator):
        self.potencial = integrator.potencial
        self.integrator = integrator
        self.result_list = []
        self.row_list = []
    
    '''
        xstart : start position point
        vstart : start velocity point
        (xstart, vstart) : is upper left corner of generated image data
        size: size of image in (x, v) units
    '''
    def calculate_basins(self, xstart, vstart, size): 
        xmin = xstart
        xmax = xstart + size
        vmin = vstart - size
        vmax = vstart

        scale = size/640.0  # after modification change grid & block sizes in GPU !!!

        self.x_array = numpy.arange(xmin, xmax, scale)
        self.v_array = numpy.arange(vmin, vmax, scale)

class GeneratorCPU(Generator):

    def calculate_basins(self, xstart, vstart, size, sim_time): 
        Generator.calculate_basins(self, xstart, vstart, size)
        
        for vel0 in self.v_array:        
            self.row_list = []                       
            
            print "v0 = ",vel0
            
            for pos0 in self.x_array:
                #print "( x0, v0 ) = ( ",pos0,", ", vel0," )"
                
                if (pos0 == 0) and (vel0 == 0):
                    self.row_list.append(1)
                else:
                    self.calculate_trajectory(pos0, vel0, sim_time)
                    
            self.result_list.append(self.row_list)

    def calculate_trajectory(self, pos0, vel0, sim_time):
        pos = pos0
        vel = vel0

        time = 0
        while (sim_time > time):
            vect = self.integrator.calculate_step(pos, vel)
            
            #print vect,self.integrator.delta_time
            
            pos = vect[0]
            vel = vect[1]
         
            time += self.integrator.delta_time

        basin = self.potencial.determine_minimum(pos)
        self.row_list.append(basin)
   
    
class GeneratorGPU(Generator):    
    
    def __init__(self, integrator):
        Generator.__init__(self, integrator)
        
        self.cuda_device_number = 0
        self.cuda_context = None
        
        self.gpu_source = integrator.potencial.gpu_source + integrator.gpu_source + """
            __global__ void basins(float *cudaResult, float *pos0, float *vel0) {
                const int idx = blockIdx.y  * gridDim.x  * blockDim.z * blockDim.y * blockDim.x + 
                                blockIdx.x  * blockDim.z * blockDim.y * blockDim.x + 
                                threadIdx.z * blockDim.y * blockDim.x + 
                                threadIdx.y * blockDim.x + 
                                threadIdx.x;

                float x = pos0[idx];
                float v = vel0[idx];
                float t = 0;
                
                do {                
                    calculateStep(t, x, v);
                    t += dt;  
                } while (t <= simTime);

                cudaResult[idx] = determineMinimum(x);
            }
        """ 

    def calculate_basins(self, xstart, vstart, size, sim_time): 
        Generator.calculate_basins(self, xstart, vstart, size)

        self.result_list = []
        pos0 = []
        vel0 = []

        for i in self.v_array:
            for j in self.x_array:
                pos0.append(i)
                vel0.append(j)
        
        pos0 = numpy.array(pos0).astype(numpy.float32)
        vel0 = numpy.array(vel0).astype(numpy.float32)
                
        self.__do_cuda_calculation(pos0, vel0, sim_time)
        
    def __do_cuda_calculation(self, pos0, vel0, sim_time):
        delta_time_source = """
              __const__ float simTime = %sf;              
        """ % (sim_time)

        cuda_result = numpy.zeros_like(pos0)
        
        self.__initalize_cuda()        
        mod = SourceModule(delta_time_source + self.gpu_source)
                
        do_basins = mod.get_function("basins")
        do_basins(cuda.Out(cuda_result), 
                  cuda.In(pos0), 
                  cuda.In(vel0), 
                  block = (16, 16, 1), 
                  grid = (40, 40))
        
        self.__deactivate_cuda()        
        self.__save_data(cuda_result)
        
    def __initalize_cuda(self):
        cuda.init() #init pycuda driver
        current_dev = cuda.Device(self.cuda_device_number) #device we are working on
        
        self.cuda_context = current_dev.make_context() #make a working context
        self.cuda_context.push() #let context make the lead
    
    def __deactivate_cuda(self):  
        self.cuda_context.pop() #deactivate again
        self.cuda_context.detach() #delete it
                
    def __save_data(self, cuda_result):   
        index = 0
        for i in self.v_array:  
            self.row_list = []
            
            for j in self.x_array:
                self.row_list.append(cuda_result[index])  
                index += 1  
                  
            self.result_list.append(self.row_list)


            
from Potencial import PotencialSimpleFactory
from BasinsGenerator import GeneratorSimpleFactory
from Graphics import ImageGenerator
import numpy


def main():
    """
        PotencialSimpleFactory.create_potencial(POTENCIAL, DEVICE, GAMMA, DELTA)
        
            POTENCIAL : potencial name
            DEVICE : CPU or GPU
            GAMMA : friction factor
            DELTA : stop parameter
            
        GeneratorSimpleFactory.createGenerator(potencial, INTEGRATOR)
        
            potencial : instance variable of PotencialSimpleFactory
            INTEGRATOR : Euler or RK4 
            
    """
    potencial = PotencialSimpleFactory.create_potencial("Potencial1", "GPU", 0.1)
    generator = GeneratorSimpleFactory.createGenerator(potencial, "RK4")
    
    print " > Calculating basins"
    generator.calculate_basins(-1, 1, 10, 20.0)

    print " > Generating image"
    ImageGenerator.generate_image(generator.result_list, "out", "PNG")


#    for item in range(0, 2000, 1):
#        delta_time = item/10.0
#        
#        if item < 10:
#            prefix = "0000"
#        elif item >= 10 and item < 100:
#            prefix = "000"
#        elif item >= 100 and item < 1000:
#            prefix = "00" 
#        elif item >= 1000 and item < 10000:
#            prefix = "0"
#            
#        file_name = prefix + str(item)
#        
#        print " > Calculating basins for",delta_time
#        generator.calculate_basins(-5, 5, 10, delta_time)
#
#        print " > Generating image"
#        ImageGenerator.generate_image(generator.result_list, file_name, "PNG")


if __name__ == "__main__":
    main()
    
    
    
    
    
    
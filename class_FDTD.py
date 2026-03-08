import numpy as np

import matplotlib.pyplot as plt

mu_vacuum_scalar_SI = np.pi*4E-7
epsilon_vacuum_scalar_SI = 8.854E-12
Speed_Of_Light=1/ np.sqrt(mu_vacuum_scalar_SI*epsilon_vacuum_scalar_SI)

mm_to_m_factor=0.001

class FDTD_2D_TM:
    def __init__(self,Length_x_Domain_mm:float,Length_y_Domain_mm:float):

        self.Length_x_Domain_m = mm_to_m_factor*Length_x_Domain_mm
        self.Length_y_Domain_m = mm_to_m_factor*Length_y_Domain_mm
        
        self.Is_Full_Automatic = False

        
        self.initialize()
        print("starting simulation")
    
    def initialize(self):
        self.Full_Automatic()
        self.initialize_grids()
        self.initialize_Time_simulation
        self.initialize_Fields()
        self.initialize_PML()

    def Full_Automatic(self):
        """
        0: full automatic. Never asks again
        1: asks each time if you want to do it manual or automatic
        """
        self.Is_Full_Automatic=bool(input("1: full automatic. Never asks again\n" \
                                "0: asks each time if you want to do it manual or automatic"))

    def Choose_Automatic_or_Manual(self):
        """
        asks the user if he wants to choose the inputs or let the program use its predetermined
        
        """

        if self.Is_Full_Automatic== True:
            self.Is_Automatic=True
            self.Is_Manual = False
            return None
        self.Is_Automatic = bool(input("1. program chooses parameters.\n 0. you choose parameters"))
        self.Is_Manual = not self.Is_Automatic
    
    def Choose_FDTD_Method(self):
        self.FDTD_Method= input("put in your method of FDTD: 0. Yee method\n" \
                                "1. Fully_collocated ")
        self.Yee = "0"
        self.Fully_Collocated = "1"
    
    #To-Do we need to create a YEE mode and a collocated mode.
    # 
    
    def initialize_grids(self):
        """

        This only works for square grids now

        there are different grids that we work it. It is of utmost importance to make sure that the fields are on their
        respective grid cells.  

        the gridcells will be defined as such:
        
        YEE:Each square: the center of that grid cell equals the Ez value
                     The edges of that grid cell equals respectively HY and HX values

        if we decide to switch this. This means that masking of the other media should switch for different formulae

        with this definition we can see if the values are in or not in the media
        
        TLDR:
        Ez is center gridcells
        HX,HY are edges gridcells

        Arrays are made for computations
        Meshes are made for plotting or for working with masks
        """

        #TO-DO when we switch to refinment we need to make sure that it is implemented very smoothly

        print("initializing grids")
        self.Choose_Automatic_or_Manual()

        if self.Is_Automatic == True:
            self.Number_Of_X_Partitions = 250
            self.Number_Of_Y_Partitions = 250

        if self.Is_Manual == True:
            self.Number_Of_X_Partitions = int( input( "How many partitions on the x-axis" ) )
            self.Number_Of_Y_Partitions = int( input( "How many partitions on the y-axis" ) )
            
        #arrays

        self.corner_Coordinates_X_array_m = np.linspace(
            0, self.Length_x_Domain_m, self.Number_Of_X_Partitions
        )  # dim (Nx,)

        self.corner_Coordinates_Y_array_m = np.linspace(
            0, self.Length_y_Domain_m, self.Number_Of_Y_Partitions
        )  # dim (Ny,)

        self.step_X_Array_m = np.diff(self.corner_Coordinates_X_array_m)  # (Nx-1,)
        self.step_Y_Array_m = np.diff(self.corner_Coordinates_Y_array_m)  # (Ny-1,)

        self.horizontal_edges_coord_X_Array_m = (
            self.corner_Coordinates_X_array_m[:-1] 
          + self.step_X_Array_m / 2
        )  # (Nx-1,)
        self.horizontal_edges_coord_Y_Array_m = self.corner_Coordinates_Y_array_m  # (Ny,)

        self.vertical_edges_coord_X_Array_m = self.corner_Coordinates_X_array_m  # (Nx,)
        self.vertical_edges_coord_Y_Array_m = (
            self.corner_Coordinates_Y_array_m[:-1] 
          + self.step_Y_Array_m / 2
        )  # (Ny-1,)

        self.center_coord_X_Array_m = self.horizontal_edges_coord_X_Array_m # (Nx-1, )
        self.center_coord_Y_Array_m = self.vertical_edges_coord_Y_Array_m   # (Ny-1, )
        self.center_step_X_Array_m  = np.diff(self.center_coord_X_Array_m)  # (Nx-2, )
        self.center_step_Y_Array_m  = np.diff(self.center_coord_Y_Array_m)  # (Nx-2, )
        
        #Meshes

        self.corner_X_mesh_cell_m, self.corner_Y_mesh_cell_m = np.meshgrid(
            self.corner_Coordinates_X_array_m,
            self.corner_Coordinates_Y_array_m,
            indexing='ij'
        )  #  (Nx,Ny)

        self.step_X_mesh_cell_m = (
            self.corner_X_mesh_cell_m[1:, :] 
          - self.corner_X_mesh_cell_m[:-1, :]
        )  #  (Nx-1,Ny)

        self.step_Y_mesh_cell_m = (
            self.corner_Y_mesh_cell_m[:, 1:] 
          - self.corner_Y_mesh_cell_m[:, :-1]
        )  #  (Nx,Ny-1)

        self.horizontal_edges_coord_X_mesh_cell_m, self.horizontal_edges_coord_Y_mesh_cell_m = np.meshgrid(
            self.horizontal_edges_coord_X_Array_m,
            self.horizontal_edges_coord_Y_Array_m,
            indexing='ij'
        ) # ( Nx-1, Ny )

        self.vertical_edges_coord_X_mesh_cell_m, self.vertical_edges_coord_Y_mesh_cell_m = np.meshgrid(
            self.vertical_edges_coord_X_Array_m,
            self.vertical_edges_coord_Y_Array_m,
            indexing='ij'
        ) # ( Nx, Ny-1 )

        self.center_coord_X_mesh_cell_m,self.center_coord_Y_mesh_cell_m  = np.meshgrid(
            self.center_coord_X_Array_m,
            self.center_coord_Y_Array_m,
            indexing='ij'
        )  # ( Nx-1, Ny-1 )

        self.center_step_X_mesh_cell_m = (
            self.center_coord_X_mesh_cell_m[1:, :] 
            - self.center_coord_X_mesh_cell_m[:-1, :]
        )  # dim = ( Nx-2, Ny-1 )

        self.center_step_Y_mesh_cell_m = (
            self.center_coord_Y_mesh_cell_m[:, 1:] 
          - self.center_coord_Y_mesh_cell_m[:, :-1]
        )  # dim = ( Nx-1, Ny-2 )

                    


    def initialize_Time_simulation(self):
        print("initializing Time_simulation")

        Minimal_Spatial_Step_X_m = min( [ min(self.step_X_Array_m), min(self.center_step_X_Array_m) ] )
        Minimal_Spatial_Step_Y_m = min( [ min(self.step_Y_Array_m), min(self.center_step_Y_Array_m) ] )


        self.Choose_Automatic_or_Manual()

        if self.Is_Automatic==True:
            self.Number_Of_Time_partitions = 500
            self.Courant_number = 1

        if self.Is_Manual==True:
            self.Number_Of_Time_partitions = int( input(" input the amount of timeframes that the simulation will run over( Time partitions)") )
            self.Courant_number = int( input(" input The Courant") )

        self.Time_Step_s = self.Stability_Condition_time_FDTD(Minimal_Spatial_Step_X_m,
                                                            Minimal_Spatial_Step_Y_m,
                                                            self.Courant_number)
        

        
        self.Total_Time_Simulation_s = self.Time_Step_s * self.Number_Of_Time_partitions
        print(f"the simulation simulates over a period of{self.Total_Time_Simulation_s} s")

    

    def initialize_Fields(self):
        print("initializing fields")
        self.Choose_FDTD_Method()
        
        if self.FDTD_Method==self.Yee:
            self.E_Field_Z = np.zeros( np.shape( self.center_coord_X_mesh_cell_m ) )            #dim ( Nx-1, Ny-1 )
            self.H_Field_X = np.zeros( np.shape( self.horizontal_edges_coord_X_mesh_cell_m ) )  #dim ( Nx-1, Ny )
            self.H_Field_Y = np.zeros( np.shape(self.vertical_edges_coord_X_mesh_cell_m ) )     #dim ( Nx, Ny-1 )
            self.J_Field_Z = np.zeros( np.shape( self.center_coord_X_mesh_cell_m ) )            #dim ( Nx-1, Ny-1 )

        if self.FDTD_Method==self.Fully_Collocated:
            pass




    def initialize_PML(self,Choose_input_type="automatic"):
        """
        sigma is the imaginary_stretch_parameter which absorbs evanescent waves
        kappa is the real_stretch_parameter which absorbs traveling waves

        Choose_input_Type: Automatic-- it does everything automatic
                           Manual -- This function asks you to input values

        Course Page 33 for formulae
        
        """

        # TO-DO make sure that PML understand the x and y-directions
        # Idea store all the fields in one dictionary and their respective grids in a list next to them as index
        # this way it is faster to call the functions
        print("initializing PML")
        self.PML_Active = 1
        if Choose_input_type =="automatic":
            self.PML_number_of_layers= 15
            self.PML_power= 4
            self.PML_real_stretch_parameter_max= 1.0  
             

        if Choose_input_type == "manual":
            self.PML_number_of_layers= input(f"Amount of layers of PML: Recommended value is 10")
            self.PML_power= input(f"The power of formula 2.131: Recommended value is 4")
            self.PML_real_stretch_parameter_max= input(f"The max value of the real stretch Parameters")

        self.PML_imaginary_stretch_parameter_max=...#(power+1)/150 pi spatial_step

        # Look for a 
        self.PML_imaginary_stretch_parameter = self.PML_imaginary_stretch_parameter_max 

    def Stability_Condition_time_FDTD( dx, dy, CFL ):
        return CFL/np.sqrt(1/dx**2+1/dy**2)
    

    def Yee_update_
        


simulation = FDTD_2D_TM(10,10) 

plt.imshow(simulation.E_Field_Z)

plt.show()

#TO-DO right now
#create Data structures for sources (DONE)
#create Geometry
#create update_schemes_with current_source
#create update_scheme with conducting theme

# Er zijn dus verschillende update_schemes. 
# 1. in vacuum
# 2. in scattere
# 3. in PML. 


#animate schemes
#choose source coordinates.
#choose observer coordinates
#find 
#create PML layer
#think of other ways
#need to look of averaging material properties in staggering grid
#think of changing it such that there are gridvoltages,left



#quality of life change ideas

#full automatic and automatic can be made better
#Idea maybe store fieldshapes in values instead of calling large data.
#Idea all the fully automatic can be stoered in FDTD_2D_TM


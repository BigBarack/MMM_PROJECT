import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
#convention E-field is normal grid(spacetime) and H-field is dual grid(spacetime)
#Current Jz= normal grid(space) and dual grid(time). 
#Such that interpolation is not needed the TM (4 ) 
#equation in the project assignment

c= 3E8
_x,_y = 0,1
Number_Of_X_Partitions = 250
Number_Of_Y_Partitions = 250
Number_Of_Time_partitions = 500
# define first the grid_cells_width such that CFL=1
geometry_len_X_meters=1
geometry_len_Y_meters=1

X_array = np.linspace(0,geometry_len_X_meters,Number_Of_X_Partitions)
Y_array = np.linspace(0,geometry_len_Y_meters,Number_Of_Y_Partitions)

Xstep_m_scalar = X_array[1]-X_array[0]
Ystep_m_scalar = Y_array[1]-Y_array[0]

dual_Xstep_m_scalar = Xstep_m_scalar
dual_Ystep_m_scalar = Ystep_m_scalar

Courant_number= 1

timestep = Courant_number / (c*np.sqrt( 1/Xstep_m_scalar**2 + 1/Ystep_m_scalar**2 ))

Total_time_simulation_seconds = timestep*Number_Of_Time_partitions
time_simulation_seconds = np.linspace(0,Total_time_simulation_seconds,Number_Of_Time_partitions) 


YEE_grid_Zshape = (Number_Of_X_Partitions,Number_Of_Y_Partitions)

YEE_Dual_Grid_XShape = (Number_Of_X_Partitions,Number_Of_Y_Partitions+1)
YEE_Dual_Grid_YShape = (Number_Of_X_Partitions+1,Number_Of_Y_Partitions)

YEE_Electric_Field_Z_component = np.zeros(YEE_grid_Zshape)

YEE_Magnetic_Field_X_component = np.zeros(YEE_Dual_Grid_XShape)
YEE_Magnetic_Field_Y_component = np.zeros(YEE_Dual_Grid_YShape)

YEE_current_Field_Z_Component = np.zeros(YEE_grid_Zshape)

Xmesh, Ymesh = np.meshgrid(X_array, Y_array, indexing="ij")


mu_vacuum_scalar_SI = np.pi*4E-7
epsilon_vacuum_scalar_SI = 8.854E-12

conductance_material_scalar = ... #To-DO only matters in material
relative_mu_scalar = ... #To-DO only matters in material
relative_epsilon_scalar = ...#To-DO only matters in material

source_Term = lambda time:np.exp(-((time-20*timestep)/(20*timestep))**2)


def centraldifference_Y(field,grid):
    #dual field and grid  --->  returns normal field and grid
    #normal field and grid ---> returns dual field and grid
    return (field[:,1:]-field[:,:-1])/grid

def centraldifference_X(field,grid):
    #dual field and grid  --->  returns normal field and grid
    #normal field and grid ---> returns dual field and grid
    return (field[1:,:]-field[:-1,:])/grid

def Yee_2D_update_equations_Full():
    """
    This is with equation
    
    """
    
    # we assume fieldz is normal grid and fieldx and fieldy dualgrid

    fieldz += (
                timestep/epsilon_vacuum_scalar_SI
                *(centraldifference_X(fieldy,dualgridx)
                -centraldifference_Y(fieldx,dualgridy)
                -sourcefield)
)


    fieldx[:,1:-1] += (
                
                        -timestep/mu_vacuum_scalar_SI
                        *centraldifference_Y(fieldz,gridy)
    )

    fieldy[1:-1,:] += (
                
                        timestep/mu_vacuum_scalar_SI
                        *centraldifference_X(fieldz,gridx)
    )

    periodic_boundary_condition(fieldx)
    periodic_boundary_condition(fieldy)
    periodic_boundary_condition(fieldz)

def periodic_boundary_condition(_2Dfield):
    _2Dfield[0,:]=_2Dfield[-1,:]
    _2Dfield[:,-1]=_2Dfield[:,0]
    

def update_source_point(field,timestep):
    shapefield=np.shape(field)
    field[ shapefield[_x]//2, shapefield[_y]//2] = source_Term( timestep )
    print(source_Term(timestep))
    

def YEE_iterate_Simulation(time_simulation,
                           fieldx,fieldy,
                           fieldz,sourcefield,
                           gridx=1,gridy=1,
                           dualgridx=1,dualgridy=1):
    fig,ax = plt.subplots()
    ax.set_xlabel('(m)')
    ax.set_ylabel('(m)')

    
    plt.axis('equal')
    pcm = ax.pcolormesh(Xmesh, Ymesh, fieldz, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax)
    movie = []
    
    for time_index,time_value in enumerate(time_simulation):
        print(fieldz)
        sourcefield.fill(0)
        update_source_point(sourcefield,time_value)
        Yee_2D_update_equations_Vacuum_(fieldx,
                                        fieldy,
                                        fieldz,
                                        sourcefield,
                                        gridx,
                                        gridy,
                                        dualgridx,
                                        dualgridy,
                                        timestep)
        frame = [ax.pcolormesh(Xmesh, Ymesh, fieldz, shading="auto")]
        movie.append(frame)

    my_anim =ArtistAnimation(fig,movie,interval=1,repeat_delay=1000)
    my_anim.save(filename='insane.gif', writer='pillow')
    
    plt.show()



YEE_iterate_Simulation(time_simulation_seconds
                       ,YEE_Magnetic_Field_X_component
                       ,YEE_Magnetic_Field_Y_component
                       ,YEE_Electric_Field_Z_component
                       ,YEE_current_Field_Z_Component
                       ,Xstep_m_scalar
                       ,Ystep_m_scalar
                       ,dual_Xstep_m_scalar
                       ,dual_Ystep_m_scalar)






#TO-DO right now
#create Data structures for sources
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

"""
Autoer: Archetti Ivan
Date: 10/01/2025

Esempi di studio profili di moto (Trapezoidal velocity profile)
"""


from MotionProfile import MotionProfile
from PlotMotionProfile import PlotMotionProfile
from TracePath import TracePath
import numpy as np
from numpy import pi

# =======================================================================================================================

def straight_line():
    # definizione punti
    start_point = (1,1)
    end_point = (2,3)

    n_points = 1000

    um1 = ("Tempo", "Spazio", "Velocità", "Accelerazione")    
    MP = MotionProfile()
    PlotMP = PlotMotionProfile(um_axes=um1)
    TrPath = TracePath()

    # discretizzo il percorso
    pos_x, pos_y = TrPath.trace_line(start_point, end_point, n_points=n_points)
    
    # mostro il percorso
    um = ("[m]", "[m]")
    PlotMP.plot_path(x=pos_x, y=pos_y, um=um)

    # calcolo la legge di moto e mostro i grafici
    Ds_x = end_point[0] - start_point[0]
    Ds_y = end_point[1] - start_point[1]
    Ds = (Ds_x**2 + Ds_y**2)**(1/2)
    Dt = 1
    t_i = 0
    s0 = 0
    shape = [0.2, 0.6, 0.2]

    # spostamento in modulo
    t, s, v, a  = MP.trapezoidal_MP(Ds=Ds, Dt=Dt, ti=t_i, s0=s0, n_points=n_points, shape=shape)

    um_x = ("[s]", "[m]", "$[\\frac{m}{s}]$", "$[\\frac{m}{s^{2}}]$")
    title_x = ["          Moto", "Spazio", "Velocità", "Accelerazione"] 
    PlotMP.plot_motion_profile(title=title_x, t=t, s=s, v=v, a=a, amax=(10,-10), vmax=(2,1), um=um_x)

    # spostamento, velocità ed accelerazione in x e y
    angle = MP.get_line_angle(P0=start_point, P1=end_point)
    sx, sy = MP.get_line_position(s=s, angle=angle)
    vx, vy = MP.get_line_speed(v=v, angle=angle)
    ax, ay = MP.get_line_acceleration(a=a, angle=angle)

    um_x = ("[s]", "[m]", "$[\\frac{m}{s}]$", "$[\\frac{m}{s^{2}}]$")
    title_x = ["          Moto in x", "Spazio in x", "Velocità in x", "Accelerazione in x"] 
    PlotMP.plot_motion_profile(title=title_x, t=t, s=sx, v=vx, a=ax, amax=(0,0), vmax=(0,0), um=um_x)
    
    um_y = ("[s]", "[m]", "$[\\frac{m}{s}]$", "$[\\frac{m}{s^{2}}]$")
    title_y = ["          Moto in y", "Spazio in y", "Velocità in y", "Accelerazione in y"] 
    PlotMP.plot_motion_profile(title=title_y, t=t, s=sy, v=vy, a=ay, amax=(0,0), vmax=(0,0), um=um_y)

# =======================================================================================================================

def circle_arc():
    # parametri della curva
    R = 1
    center = (0, 0)
    starting_angle = 0
    angle = pi/2
    
    n_points = 1000

    um1 = ("Tempo", "Spazio", "Velocità", "Accelerazione")    
    MP = MotionProfile()
    PlotMP = PlotMotionProfile(um_axes=um1)
    TrPath = TracePath()

    # discretizzo il percorso
    pos_x, pos_y = TrPath.trace_arc(radius=R, center=center, start_angle=starting_angle, angle=angle, n_points=n_points)
    
    # mostro il percorso
    um = ("[m]", "[m]")
    PlotMP.plot_path(x=pos_x, y=pos_y, um=um)

    # calcolo la legge di moto
    Dt = 1
    ti = 0
    shape = [0.2, 0.6, 0.2]
    s0 = starting_angle*R
    Ds = angle*R
    kin_param = list(MP.trapezoidal_MP(Ds=Ds, Dt=Dt, ti=ti, s0=s0, n_points=n_points, shape=shape))
    t, s, v, a, at, ac = MP.arc_motion(kin_param, radius=R, start_angle=starting_angle)

    um1 = ["[s]", "[m]", "$[\\frac{m}{s}]$", "$[\\frac{m}{s^{2}}]$"]
    title = ["          Moto curvilineo", "Spazio", "Velocità", "Accelerazione totale"]
    PlotMP.plot_motion_profile(title=title, t=t, s=s, v=v, a=a, um=um1)

    # spostamento, velocità ed accelerazione in x e y
    sx, sy = MP.get_arc_position()
    vx, vy = MP.get_arc_speed()
    atx, aty = MP.get_arc_tan_acceleration()
    acx, acy = MP.get_arc_centr_acceleration()
    ax, ay = MP.get_arc_total_acceleration()

    # mostro l'accelerazione centripeta
    PlotMP.plot_kinematic_value(title="Accelerazione centripeta", t=t, x=ac, um=["s","$[\\frac{m}{s^{2}}]$"], 
                                label=["tempo", "Accelerazione centripeta"], color="RED")
    
    # mostro l'accelerazione tangenziale
    PlotMP.plot_kinematic_value(title="Accelerazione tangenziale", t=t, x=at, um=["s","$[\\frac{m}{s^{2}}]$"], 
                                label=["tempo", "Accelerazione tangenziale"], color="RED")

    # mostro l'accelerazione totale
    PlotMP.plot_kinematic_value(title="Accelerazione totale", t=t, x=a, um=["s","$[\\frac{m}{s^{2}}]$"], 
                                label=["tempo", "Accelerazione totale"], color="RED")
    
    # cinematica proiettata sull'asse x e y
    um_x = ("[s]", "[m]", "$[\\frac{m}{s}]$", "$[\\frac{m}{s^{2}}]$")
    title_x = ["          Moto in x", "Spazio in x", "Velocità in x", "Accelerazione totale in x"] 
    PlotMP.plot_motion_profile(title=title_x, t=t, s=sx, v=vx, a=ax, amax=(0,0), vmax=(0,0), um=um_x)
    
    um_y = ("[s]", "[m]", "$[\\frac{m}{s}]$", "$[\\frac{m}{s^{2}}]$")
    title_y = ["          Moto in y", "Spazio in y", "Velocità in y", "Accelerazione in y"] 
    PlotMP.plot_motion_profile(title=title_y, t=t, s=sy, v=vy, a=ay, amax=(0,0), vmax=(0,0), um=um_y)


# =======================================================================================================================

def closed_loop():
    # prove legge di moto
    um1 = ("Tempo", "Spazio", "Velocità", "Accelerazione")    
    MP = MotionProfile()
    PlotMP = PlotMotionProfile(um_axes=um1)
    TrPath = TracePath()

    n_points = 1000

    # geometrie del percorso
    h = 1
    w = 2
    R = 0.4

    pos_x, pos_y = TrPath.trace_rounded_rectangle(center=(0,0), rectangle_angle=0, fillet_radius=R, heigth=h, width=w, n_points=n_points)

    # mostro il percorso
    um2 = ("[m]", "[m]")
    PlotMP.plot_path(x=pos_x, y=pos_y, um=um2)
 
    # creo la cinematica
    t = np.array([])
    s = np.array([])
    v = np.array([])
    a = np.array([])

    t0 = 0
    s0 = 0

    v_max = 1

    shape_acc = [0.2, 0.8, 0] 
    shape_cost = [0, 1, 0]     
    shape_dec = [0, 0.8, 0.2]
    shapes = [x for x in [shape_acc] + [shape_cost] * 6 + [shape_dec]]
    
    p_len = TrPath.get_rounded_rectangle_perimeter(fillet_radius=R, heigth=h, width=w)

    for i, (Ds, shape) in enumerate(zip(p_len, shapes)):       
        Dt = Ds/(v_max*(shape[0]/2 + shape[1] + shape[2]/2))

        # applico la LDM trapezoidale distinguendo tratti dritti e curvi
        ti, si, vi, ai = MP.trapezoidal_MP(Ds=Ds, Dt=Dt, ti=t0, s0=s0, n_points=n_points, shape=shape)
        
        if i % 2 == 1:
            kin_param = ti, si, vi, ai
            ti, si, vi, ai, ait, aic = MP.arc_motion(kin_param, radius=R, start_angle=0)

        s0 = si[-1]
        t0 = ti[-1]

        t = np.concatenate([t, ti])
        s = np.concatenate([s, si])
        v = np.concatenate([v, vi])
        a = np.concatenate([a, ai])
    
    # traccio la legge di moto complessiva
    um1 = ["[s]", "[m]", "$[\\frac{m}{s}]$", "$[\\frac{m}{s^{2}}]$"]
    title = ["          Moto rettilineo", "Spazio percorso", "Velocità percorso", "Accelerazione totale"]
    PlotMP.plot_motion_profile(title=title, t=t, s=s, v=v, a=a, um=um1)

# =======================================================================================================================

def tool_path():
    # prove legge di moto
    um1 = ("Tempo", "Spazio", "Velocità", "Accelerazione")    
    MP = MotionProfile()
    PlotMP = PlotMotionProfile(um_axes=um1)
    TrPath = TracePath()

    n_points = 1000

    # geometrie del percorso
    w = 10
    h = 20
    n_step = 10

    pos_x, pos_y = TrPath.trace_milling_path(P_start=(0,0), n_step=n_step, height=h, width=w, n_points=n_points)

    # mostro il percorso
    um2 = ("[m]", "[m]")
    PlotMP.plot_path(x=pos_x, y=pos_y, um=um2)

    # creo la cinematica
    t = np.array([])
    s = np.array([])
    v = np.array([])
    a = np.array([])

    t0 = 0
    s0 = 0

    v_max = 1

    shape_acc = [0.2, 0.8, 0] 
    shape_cost = [0, 1, 0]     
    shape_dec = [0, 0.8, 0.2]
    shapes = [x for x in [shape_acc] + [shape_cost] * (2*n_step-1) + [shape_dec]]

    R = h/(2*n_step)

    p_len = TrPath.get_milling_path_perimeter(n_step=n_step, heigth=h, width=w)

    for i, (Ds, shape) in enumerate(zip(p_len, shapes)):       
        Dt = Ds/(v_max*(shape[0]/2 + shape[1] + shape[2]/2))

        # applico la LDM trapezoidale distinguendo tratti dritti e curvi
        ti, si, vi, ai = MP.trapezoidal_MP(Ds=Ds, Dt=Dt, ti=t0, s0=s0, n_points=n_points, shape=shape)
        
        if i % 2 == 1:
            kin_param = ti, si, vi, ai
            ti, si, vi, ai, ait, aic = MP.arc_motion(kin_param, radius=R, start_angle=0)

        s0 = si[-1]
        t0 = ti[-1]

        t = np.concatenate([t, ti])
        s = np.concatenate([s, si])
        v = np.concatenate([v, vi])
        a = np.concatenate([a, ai])
    
    # traccio la legge di moto complessiva
    um1 = ["[s]", "[m]", "$[\\frac{m}{s}]$", "$[\\frac{m}{s^{2}}]$"]
    title = ["          Moto rettilineo", "Spazio percorso", "Velocità percorso", "Accelerazione totale"]
    PlotMP.plot_motion_profile(title=title, t=t, s=s, v=v, a=a, um=um1)
    

# =======================================================================================================================

def poligonon_loop():
    um1 = ("Tempo", "Spazio", "Velocità", "Accelerazione")
    MP = MotionProfile()
    PlotMP = PlotMotionProfile(um_axes=um1)
    TrPath = TracePath()

    n_points = 1000
    N = 6
    r = 0.4
    R = 1
    p_angle = 0

    pos_x, pos_y = TrPath.trace_regular_rounded_polygon(center=(0,0), n_sides=N, fillet_radius=r, radius=R, polygon_angle=p_angle, n_points=n_points)
    um2 = ("[m]", "[m]")
    PlotMP.plot_path(x=pos_x, y=pos_y, um=um2)

    # calcolo le geometrie di supporto
    beta = 2*pi/N                         # angolo interno raccordo
    l_arco = beta*r                       # arco di raccordo radrizzato

    h = R*np.sqrt(1-(np.sin(pi/N))**2)    # distanza tra il centro del poligono e il centro del lato
    hp = h - r                            # distanza tra il centro del poligono e il centro del lato meno il raggio di raccordo
    Lp = 2*hp*np.tan(beta/2)              # lunghezza lato poligono senza raccordo

    # creo la cinematica
    t = np.array([])
    s = np.array([])
    v = np.array([])
    a = np.array([])

    t0 = 0
    s0 = 0

    v_max = 1

    shape_acc = [0.2, 0.8, 0] 
    shape_cost = [0, 1, 0]     
    shape_dec = [0, 0.8, 0.2]
    shapes = [x for x in [shape_acc] + [shape_cost] * 2*(N-1) + [shape_dec]]

    for i, shape in enumerate(shapes):       
        if i % 2 == 0:
            Ds = Lp
        else:
            Ds = l_arco

        Dt = Ds/(v_max*(shape[0]/2 + shape[1] + shape[2]/2))

        # applico la LDM trapezoidale distinguendo tratti dritti e curvi
        ti, si, vi, ai = MP.trapezoidal_MP(Ds=Ds, Dt=Dt, ti=t0, s0=s0, n_points=n_points, shape=shape)
        
        if i % 2 == 1:
            kin_param = ti, si, vi, ai
            ti, si, vi, ai, ait, aic = MP.arc_motion(kin_param, radius=R, start_angle=0)

        s0 = si[-1]
        t0 = ti[-1]

        t = np.concatenate([t, ti])
        s = np.concatenate([s, si])
        v = np.concatenate([v, vi])
        a = np.concatenate([a, ai])
    
    # traccio la legge di moto complessiva
    um1 = ["[s]", "[m]", "$[\\frac{m}{s}]$", "$[\\frac{m}{s^{2}}]$"]
    title = ["          Moto rettilineo", "Spazio percorso", "Velocità percorso", "Accelerazione totale"]
    PlotMP.plot_motion_profile(title=title, t=t, s=s, v=v, a=a, um=um1)

# =======================================================================================================================



if __name__ == "__main__":
    # seleziona il umero per visualizzare gli esempi
    menu = 0
    
    if menu == 0:
        straight_line()
    elif menu == 1:
        circle_arc()
    elif menu == 2:
        closed_loop()
    elif menu == 3:
        tool_path()
    elif menu == 4:
        poligonon_loop()


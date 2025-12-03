import numpy as np
from matplotlib import pyplot
from open_atmos_jupyter_utils import show_plot, show_anim
from PyMPDATA import ScalarField, Solver, Stepper, VectorField, Options, boundary_conditions
from PyMPDATA.boundary_conditions import Extrapolated
from matplotlib.colors import LinearSegmentedColormap
colors = [
    (0.00, 0.30, 0.60),  # deep ocean  
    (0.70, 0.90, 1.00),  # shallow water 
    (0.95, 0.85, 0.60),  # beach sand 
]
beach_cmap = LinearSegmentedColormap.from_list("beach", colors)
colors2 = [
    (0.95, 0.85, 0.60),  # beach sand 
    (0.70, 0.90, 1.00),  # shallow water 

    (0.00, 0.30, 0.60),  # deep ocean  
    
]
beach_cmap2 = LinearSegmentedColormap.from_list("beach", colors2)

def plot_both(frame, *, zlim=(-0.25, 0.25),output,bathymetry):
    psi = output['h'][frame] - bathymetry
    xi, yi = np.indices(psi.shape)
    fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 6))
    ax.plot_wireframe(xi+.5, yi+.5, psi, color='blue', linewidth=.5)
    ax.set(zlim=zlim, proj_type='ortho', title=f"FAZA 1+2: t/Δt = {frame}", zlabel=r"$\zeta$")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('black')
        axis.pane.set_alpha(1)
    for axis in ('x', 'y'):
        getattr(ax, f'set_{axis}label')(f"{axis} / Δ{axis}")
    pyplot.colorbar(
        ax.contourf(xi+.5, yi+.5, bathymetry, zdir='z', offset=zlim[0], cmap=beach_cmap),
        pad=.1, aspect=10, fraction=.02, label='bathymetry', location='left'
    ).ax.invert_yaxis()
    return fig

class ShallowWaterEquationsIntegrator:
    def __init__(self, *, h_initial: np.ndarray, uh_initial:np.ndarray, vh_initial:np.ndarray,bathymetry:np.ndarray, options: Options = None):
        """ initializes the solvers for a given initial condition of `h` assuming zero momenta at t=0 """
        self.bathymetry = bathymetry.copy()
        options = options or Options(nonoscillatory=True, infinite_gauge=True)
        X, Y, grid = 0, 1, h_initial.shape
        stepper = Stepper(options=options, grid=grid,n_threads=1) #klasa która odpowiada za wykonywanie kroków czasowych - kompilowane numbą, numeryczne kroki wykonaywane by uzyskać równanie adwekcji / adekcji - dyfuzji ; ,n_threads=1 - powoduje że wyłaczamy wielowątkowość
        kwargs = {
            'boundary_conditions': [boundary_conditions.Constant(value=0)] * len(grid), #len(grid) - krotka wymiarów - przemnożone to znaczy że skopiowane 
            'halo': options.n_halo,
        } 
        if uh_initial is None:
            uh_initial = np.zeros(grid)
        if vh_initial is None:
            vh_initial = np.zeros(grid)
        advectees = {
            "h": ScalarField(h_initial, **kwargs), #**kwargs - trzy razy chcemy to samo przekazać więc robimy słownik argumentów funkcji i rozpakujemy go póżniej 
            "uh": ScalarField(uh_initial, **kwargs),
            "vh": ScalarField(vh_initial, **kwargs),
        }
        self.advector = VectorField((  #pole klasy - self.advector
                np.zeros((grid[X] + 1, grid[Y])),
                np.zeros((grid[X], grid[Y] + 1))
            ), **kwargs
        )
        self.solvers = { k: Solver(stepper, v, self.advector) for k, v in advectees.items() } #generator - tworzymy nowy słownik - wygenerujemy 

    def __getitem__(self, key): # gdy wywołamy klase np a = A() to element a['coś '] - spowoduje że wywoła się ta funkcja i zwróci coś- ty element słownika   
        """ returns `key` advectee field of the current solver state """
        return self.solvers[key].advectee.get()
    
    def _apply_half_rhs(self, *, key, axis, g_times_dt_over_dxy):
        """ applies half of the source term in the given direction """
        self[key][:] -= .5 * g_times_dt_over_dxy * self['h'] * np.gradient(self['h']- self.bathymetry, axis=axis)

    def _update_courant_numbers(self, *, axis, key, mask, dt_over_dxy):
        """ computes the Courant number component from fluid column height and momenta fields """
        velocity = np.where(mask, np.nan, 0)
        momentum = self[key]
        np.divide(momentum, self['h'], where=mask, out=velocity)
        all = slice(None, None) 
        all_but_last = slice(None, -1)
        all_but_first_and_last = slice(1, -1)
        velocity_at_cell_boundaries = velocity[( 
            (all_but_last, all),
            (all, all_but_last),
        )[axis]] + np.diff(velocity, axis=axis) / 2 
        courant_number = self.advector.get_component(axis)[(
            (all_but_first_and_last, all),
            (all, all_but_first_and_last)
        )[axis]]
        courant_number[:] = velocity_at_cell_boundaries * dt_over_dxy[axis]
        assert np.amax(np.abs(courant_number)) <= 1

    def __call__(self, *, nt: int, g: float, dt_over_dxy: tuple, outfreq: int, eps: float=1e-7):  # definuiuje co się stanie jak wywołamy a() - przykładowo instancję klasy z okrągłym nawiasem 
        """ integrates `nt` timesteps and returns a dictionary of solver states recorded every `outfreq` step[s] """
        output = {k: [] for k in self.solvers.keys()}
        for it in range(nt + 1): 
            if it != 0:
                mask = self['h'] > eps
                for axis, key in enumerate(("uh", "vh")):
                    self._update_courant_numbers(axis=axis, key=key, mask=mask, dt_over_dxy=dt_over_dxy)
                self.solvers["h"].advance(n_steps=1)
                for axis, key in enumerate(("uh", "vh")):
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
                    self.solvers[key].advance(n_steps=1)
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
                shore_zone = 10                    # ostatnie 6 komórek przy brzegu
                depth = self.bathymetry[:, -shore_zone:]
                inv = depth.max() - depth          # im płycej → mocniejsze tłumienie
                scale = inv / (inv.max() + 1e-6)
                damping_strength = 0.7           # regulujesz jak bardzo woda ma wygasać
                damping = damping_strength * (np.linspace(0, 1, shore_zone)**2)
                self['uh'][:, -shore_zone:] *= (1 - damping * scale)
                self['vh'][:, -shore_zone:] *= (1 - damping * scale)
            if it % outfreq == 0:
                for key in self.solvers.keys():
                    output[key].append(self[key].copy())
        return output
   
def run_phase(h0, uh0=None, vh0=None, *, nt, g, dt_over_dxy, outfreq,bathymetry):
    """
    Run one fase of simulation ('wave is going to the beach' or 'wave returning') and return last parameters (output, h_end, uh_end, vh_end).
    """
    output =  ShallowWaterEquationsIntegrator(h_initial=h0, uh_initial=uh0, vh_initial=vh0,bathymetry=bathymetry)(nt=nt, g=g, dt_over_dxy=dt_over_dxy, outfreq=outfreq)
    h_end  = output['h'][-1]
    uh_end = output['uh'][-1]
    vh_end = output['vh'][-1]
    return output, h_end, uh_end, vh_end


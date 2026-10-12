
"""Group-IV band alignment (simple lineup).

This script computes band offsets using *absolute* band edge energies (relative to vacuum)
if they are provided (Ev_av and Ec_{ind,dir}).

Notes
-----
- If you want *strained* band alignment for a pseudomorphic film, you also need
  deformation potentials (e.g., a_c, a_v, b, d). Elastic constants alone are not enough
  to determine band-edge shifts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _save_figure(
	fig: "plt.Figure",
	out_path: str | Path,
	*,
	also_save_svg: bool = True,
	dpi: int = 200,
) -> list[Path]:
	"""Save a Matplotlib figure.

	Behavior
	--------
	- Always saves to `out_path`.
	- If `out_path` ends with `.pdf` and `also_save_svg=True`, also saves a sibling `.svg`.
	"""
	path = Path(out_path)
	path.parent.mkdir(parents=True, exist_ok=True)

	saved: list[Path] = []
	fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight")
	saved.append(path)

	if also_save_svg and path.suffix.lower() == ".pdf":
		svg_path = path.with_suffix(".svg")
		fig.savefig(svg_path, transparent=True, bbox_inches="tight")
		saved.append(svg_path)

	return saved

#%% plot color
nn_blue = (19/255, 173/255, 181/255)
violet = (212/255, 0/255, 85/255)

#%% font settings for plots
plt.rcParams['font.sans-serif'] = ['Arial']  # Arial for common text
plt.rcParams['mathtext.fontset'] = 'custom'  
plt.rcParams['mathtext.rm'] = 'Arial'        # Arial for tex mode
plt.rcParams['mathtext.it'] = 'Arial:italic' # Arial for tex mode
plt.rcParams['mathtext.bf'] = 'Arial:bold'   # Arial for tex mode

def _linspace(a: float, b: float, n: int) -> list[float]:
	if n < 2:
		raise ValueError("n must be >= 2")
	step = (b - a) / (n - 1)
	return [a + i * step for i in range(n)]


def _varshni_delta(alpha: float, beta: float, T: float) -> float:
	"""Return Varshni gap reduction term: alpha*T^2/(T+beta).

	If alpha<=0 or beta<=0, returns 0.
	"""
	if alpha <= 0 or beta <= 0:
		return 0.0
	if T < 0:
		raise ValueError(f"Temperature must be >= 0 K. Got: {T}")
	return alpha * (T * T) / (T + beta)


def apply_temperature_to_params(
	params: dict[str, float],
	*,
	temperature_K: float,
	reference_temperature_K: float,
) -> dict[str, float]:
	"""Apply temperature dependence to Eg_ind/Eg_dir using Varshni parameters.

	This treats existing Eg values in MATERIALS as being valid at reference_temperature_K
	and adjusts them to temperature_K by:
	  Eg(T) = Eg(Tref) - Δ(T) + Δ(Tref)
	where Δ(T)=alpha*T^2/(T+beta).

	If Varshni parameters are missing, the corresponding Eg is left unchanged.
	Ec_ind/Ec_dir are recomputed from VBM + Eg to keep internal consistency.
	"""
	if temperature_K == reference_temperature_K:
		return params

	out = dict(params)

	# Direct gap (Gamma)
	if (
		"Eg_dir" in out
		and "bandgap_alpha_gamma" in out
		and "bandgap_beta_gamma" in out
		and "Ev_av" in out
		and "delta0" in out
	):
		a = out["bandgap_alpha_gamma"]
		b = out["bandgap_beta_gamma"]
		dT = _varshni_delta(a, b, temperature_K) - _varshni_delta(a, b, reference_temperature_K)
		out["Eg_dir"] = out["Eg_dir"] - dT

	# Indirect gap (L)
	if (
		"Eg_ind" in out
		and "bandgap_alpha_l" in out
		and "bandgap_beta_l" in out
		and "Ev_av" in out
		and "delta0" in out
	):
		a = out["bandgap_alpha_l"]
		b = out["bandgap_beta_l"]
		dT = _varshni_delta(a, b, temperature_K) - _varshni_delta(a, b, reference_temperature_K)
		out["Eg_ind"] = out["Eg_ind"] - dT

	# Keep conduction edges consistent with (VBM + Eg)
	vbm = out["Ev_av"] + out["delta0"] / 3.0
	if "Eg_ind" in out:
		out["Ec_ind"] = vbm + out["Eg_ind"]
	if "Eg_dir" in out:
		out["Ec_dir"] = vbm + out["Eg_dir"]

	return out


#%% material parameters

MATERIALS: dict[str, dict[str, float]] = {
	"Ge": {
		"a": 5.658,  # lattice parameter (Å), ok
		"c11": 1.285,  # elastic constant (units as given), ok
		"c12": 0.483, # ok
		# "c44": 0.668,
		"c44": 0.683, # ok
		# "delta0": 0.30,  # spin-orbit splitting (eV)
		"delta0": 0.29,  # spin-orbit splitting (eV), ok
		"Ev_av": -6.35,  # average valence band edge relative to vacuum (eV), ok
		# "Eg_ind": 0.7412,  # band gap used in this model (eV)
		"Eg_ind": 0.74,  # band gap used in this model (eV), ok
		"Ec_ind": -5.51,  # CBM used in this model relative to vacuum (eV), ok
		# "Eg_dir": 0.8893,
		"Eg_dir": 0.89, # ok
		"Ec_dir": -5.36, # ok
		"bandgap_alpha_gamma": 6.842e-4,
		"bandgap_beta_gamma": 398,
		"bandgap_alpha_l": 0.4774e-3,
		"bandgap_beta_l": 235,
		"a_v": 1.24,  # valence band deformation potential (eV), ok
		"a_c_ind": -1.54,  # conduction band deformation potential (eV), ok
		"a_c_dir": -8.24,  # conduction band deformation potential (eV), ok
		"b": -2.9,  # valence shear deformation potential (eV), ok
	},
	"Si": {
		"a": 5.43,
		"c11": 1.675,
		"c12": 0.650,
		"c44": 0.801,
		"delta0": 0.04,
		"Ev_av": -7.03,
		"Eg_ind": 2.01,
		"Ec_ind": -5.85,
		"Eg_dir": 3.39,
		"Ec_dir": -3.65,
		"bandgap_alpha_gamma": 0.4730e-3,
		"bandgap_beta_gamma": 636,
		"bandgap_alpha_l": 0.4730e-3,
		"bandgap_beta_l": 636,
		"a_v": 2.46,  # valence band deformation potential (eV)
		"a_c_ind": 4.18,  # conduction band deformation potential (eV)
		"a_c_dir": 1.98,  # conduction band deformation potential (eV)
		"b": -2.1,  # valence shear deformation potential (eV)
	},
	"Sn": {
		"a": 6.489,  # lattice parameter (Å), ok
		"c11": 0.69,  # elastic constant (units as given), ok
		"c12": 0.293,  # ok
		"c44": 0.362,  # ok
		"delta0": 0.8,  # ok
		# alpha-Sn reference (from your snippet):
		#   VBM(Sn) is +0.69 eV above VBM(Ge) [Jaros] (as commented in the database)
		# In this script, we approximate: VBM ≈ Ev_av + delta0/3.
		# With Ge: Ev_av=-6.35, delta0=0.30 -> VBM(Ge)=-6.25 eV.
		# So VBM(Sn)=-6.25+0.69=-5.56 eV and Ev_av(Sn)=VBM(Sn)-delta0/3=-5.8267 eV.
		# "Ev_av": -5.8267,
		"Ev_av": -5.83,  # ok
		# Indirect edge (L): bandgap = 0.093 eV (room temperature) from your snippet.
		# Using the same VBM alignment as above:
		#   VBM(Sn) = -5.56 eV  ->  Ec_ind = VBM + Eg_ind = -5.467 eV.
		# "Eg_ind": 0.093,
		"Eg_ind": 0.09,  # ok
		# "Ec_ind": -5.4670,
		"Ec_ind": -5.473,  # ok
		# "Eg_dir": -0.413,
		"Eg_dir": -0.41,  # ok
		# Conduction edge (Gamma, direct) consistent with Eg being defined to VBM:
		# Ec = VBM + Eg = -5.56 + (-0.413) = -5.973 eV.
		"Ec_dir": -5.9730,  # ok
		"bandgap_alpha_gamma": 0,
		"bandgap_beta_gamma": 0,
		"bandgap_alpha_l": 2.9e-4,
		"bandgap_beta_l": 0,
		"a_v": 1.58,  # valence band deformation potential (eV), ok
		"a_c_ind": -2.14,  # conduction band deformation potential (eV), ok
		"a_c_dir": -6.0,  # conduction band deformation potential (eV), ok
		"b": -2.7,  # valence shear deformation potential (eV), ok
	},	
	"Ge_tensile_sub": {
		"a": 5.664,  # lattice parameter (Å)
		"c11": 1.285,  # elastic constant (units as given), ok
		"c12": 0.483, # ok
		# "c44": 0.668,
		"c44": 0.683, # ok
		# "delta0": 0.30,  # spin-orbit splitting (eV)
		"delta0": 0.29,  # spin-orbit splitting (eV), ok
		"Ev_av": -6.35,  # average valence band edge relative to vacuum (eV), ok
		# "Eg_ind": 0.7412,  # band gap used in this model (eV)
		"Eg_ind": 0.74,  # band gap used in this model (eV), ok
		"Ec_ind": -5.51,  # CBM used in this model relative to vacuum (eV), ok
		# "Eg_dir": 0.8893,
		"Eg_dir": 0.89, # ok
		"Ec_dir": -5.36, # ok
		"bandgap_alpha_gamma": 6.842e-4,
		"bandgap_beta_gamma": 398,
		"bandgap_alpha_l": 0.4774e-3,
		"bandgap_beta_l": 235,
		"a_v": 1.24,  # valence band deformation potential (eV), ok
		"a_c_ind": -1.54,  # conduction band deformation potential (eV), ok
		"a_c_dir": -8.24,  # conduction band deformation potential (eV), ok
		"b": -2.9,  # valence shear deformation potential (eV), ok
	},
}


#%% alloy bowing parameters
#
# For a binary alloy A_x B_(1-x), a common model is
#   P(x) = (1-x) P(A) + x P(B) - b * x (1-x)
# which generalizes to multi-component alloys via pairwise terms:
#   P({x_i}) = Σ x_i P(i) - Σ_{i<j} b_ij x_i x_j
#
# Here we apply bowing only to parameters where it is physically motivated and where
# you have a known value. For GeSn, the direct gap bowing (Γ) is strong.
BOWING_PARAMETERS: dict[tuple[str, str], dict[str, float]] = {
	("Ge", "Sn"): {
		"Eg_dir": 2.46,  # eV (GeSn Γ-gap bowing)
		"Eg_ind": 0.99,  # optional: set if you have an L-gap bowing value
		# "Eg_dir": 2.46,  # eV (GeSn Γ-gap bowing)
		# "Eg_ind": 1.03,  # optional: set if you have an L-gap bowing value
		# "Eg_dir": 2.61,  # eV (GeSn Γ-gap bowing), at 10K
		# "Eg_ind": 0.80,  # optional: set if you have an L-gap bowing value, at 10K
	},
}


#%% band calculation


def _binary_alloy_composition(A: str, B: str, x_B: float) -> dict[str, float]:
	"""Return composition dict for A(1-x)B(x)."""
	if not (0.0 <= x_B <= 1.0):
		raise ValueError(f"Alloy fraction x_B must be within [0,1]. Got: {x_B}")
	return {A: 1.0 - x_B, B: x_B}


def _default_alloy_name(A: str, B: str) -> str:
	"""Human-readable alloy label, e.g. GeSn."""
	return f"{A}{B}"

substrate_name = "Ge_tensile_sub"
# Substrate alloy composition. Leave as None when using the endpoints + x_B helper below.
substrate_composition: dict[str, float] | None = None

# If you want an alloy substrate, set endpoints + x_B instead of manually writing a dict.
# Example: Ge(1-x)Sn(x) substrate with x=0.10
substrate_alloy_endpoints: tuple[str, str] | None = None
substrate_alloy_x_B: float | None = None

# Any alloy can be represented by a composition dict (linear interpolation assumed)
thin_film_name = "GeSn"
thin_film_composition: dict[str, float] | None = {"Ge": 0.9219, "Sn": 0.0781}

# Select which band edge to use for Ec/Eg in offsets & plots: "ind", "dir", or "both"
band_edge_kind = "both"

# Temperature settings for bandgaps (Varshni). Existing Eg values are treated as valid at T_ref.
temperature_K: float = 300.0
reference_temperature_K: float = 0

# If True, sweep binary alloy composition and summarize bandgaps.
do_composition_sweep: bool = False

# Sweep settings (binary alloy A(1-x)B(x))
sweep_alloy_name: str = "GeSn"  # label used in print/plot
sweep_endpoints: tuple[str, str] = ("Ge", "Sn")  # (A, B)
sweep_x_min: float = 0.0
sweep_x_max: float = 0.30
sweep_points: int = 31
sweep_plot: bool = False

# If True, plot both unstrained and strained curves on the same axes.
# - Unstrained: bulk alloy bandgaps
# - Strained: (001) biaxial strain to substrate with deformation-potential shifts
sweep_compare_strain_states: bool = False

# Add a vertical marker line on the sweep plot at a specific composition.
# Set to None to disable. Unit is percent (e.g. 8.0 -> x=0.08).
sweep_mark_x_percent: float | None = None

# If True, save the composition sweep plot to a PDF.
sweep_save_plot: bool = False
# Optional explicit save path for the sweep plot. If None and sweep_save_plot=True,
# a default name is generated next to this script.
sweep_plot_save_path: str | Path | None = False

# If True, include pseudomorphic strain-induced band-edge shifts (requires deformation potentials).
include_strain_band_shifts = True

# If True, use the composition-dependent Ge1-ySny deformation-potential model
# reported by Jiang et al. for strained-band calculations:
#   a_c_dir(y) = -9.46 + 2.33 y
#   a_c_ind(y) = -3.60 + 1.77 y
#   b(y) = -1.88 eV
# The existing MATERIALS values remain the default when this is False.
use_paper_deformation_potentials = False	 # doi:10.1088/0268-1242/29/11/115028

# Relaxation degree of the film strain.
#   0.0: fully pseudomorphic (coherent) film (max biaxial strain)
#   1.0: fully relaxed film (no biaxial strain)
# Values between 0 and 1 represent partial relaxation (fraction of mismatch relieved).
relaxation_degree: float | None = 0.573
# Which valence band defines the VBM under strain: "vbm" (=max(HH,LH)), "hh", or "lh".
strained_valence_band = "vbm"


# ---- derived configuration (keep user inputs above) ----
if substrate_alloy_endpoints is not None:
	if substrate_alloy_x_B is None:
		raise ValueError("substrate_alloy_x_B must be set when substrate_alloy_endpoints is provided")
	A_sub, B_sub = substrate_alloy_endpoints
	substrate_composition = _binary_alloy_composition(A_sub, B_sub, substrate_alloy_x_B)
	# Use a readable label for output/filenames.
	substrate_name = _default_alloy_name(A_sub, B_sub)


def _normalize_composition(composition: dict[str, float]) -> dict[str, float]:
	"""Normalize composition weights to sum to 1 (non-negative)."""
	if len(composition) == 0:
		raise ValueError("composition must not be empty")

	for k, v in composition.items():
		if v < 0:
			raise ValueError(f"composition contains negative fraction: {k}={v}")

	total = sum(composition.values())
	if total <= 0:
		raise ValueError("composition fractions must sum to a positive number")

	return {k: v / total for k, v in composition.items() if v > 0}


def make_alloy_params(composition: dict[str, float]) -> dict[str, float]:
	"""Create alloy parameters by linear interpolation of base MATERIALS.

	Parameters
	----------
	composition:
		Dict of fractions, e.g. {"Si": 0.7, "Ge": 0.3}.
		Fractions are normalized internally.
	"""
	comp = _normalize_composition(composition)

	# Validate base materials exist
	for mat in comp.keys():
		if mat not in MATERIALS:
			raise KeyError(f"Unknown base material in composition: {mat}. Available: {list(MATERIALS)}")

	# Use keys from the first material as the reference parameter set
	ref_mat = next(iter(comp.keys()))
	ref_keys = set(MATERIALS[ref_mat].keys())
	for mat in comp.keys():
		if set(MATERIALS[mat].keys()) != ref_keys:
			raise ValueError(
			"All base materials must define the same parameter keys. "
			f"Mismatch found for: {mat}"
		)

	out: dict[str, float] = {}
	for key in ref_keys:
		out[key] = sum(comp[mat] * MATERIALS[mat][key] for mat in comp.keys())

	# Apply pairwise bowing where specified.
	# Note: bowing is defined for an (unordered) material pair.
	mats = sorted(comp.keys())
	for i in range(len(mats)):
		for j in range(i + 1, len(mats)):
			mi, mj = mats[i], mats[j]
			pair = (mi, mj)
			if pair not in BOWING_PARAMETERS:
				pair = (mj, mi)
				if pair not in BOWING_PARAMETERS:
					continue
			for key, b in BOWING_PARAMETERS[pair].items():
				if key in out:
					out[key] = out[key] - b * comp[mi] * comp[mj]

	# Keep band edges consistent if Eg values were bowed.
	# In this script we treat Ec as derived from VBM + Eg so offsets/plots remain consistent.
	vbm = out["Ev_av"] + out["delta0"] / 3.0
	if "Eg_ind" in out:
		out["Ec_ind"] = vbm + out["Eg_ind"]
	if "Eg_dir" in out:
		out["Ec_dir"] = vbm + out["Eg_dir"]
	return out


def sweep_binary_alloy_bandgaps(
	A: str,
	B: str,
	*,
	name: str,
	x_min: float,
	x_max: float,
	points: int,
	plot: bool,
	mark_x: float | None = None,
	include_strain: bool = False,
	compare_strain_states: bool = False,
	relaxation_degree: float,
	substrate: str | None = None,
	substrate_composition: dict[str, float] | None = None,
	strained_valence_band: str = "vbm",
	save_path: str | Path | None = None,
	show: bool = True,
) -> None:
	"""Sweep a binary alloy A(1-x)B(x) and print/plot Eg_ind and Eg_dir.

	Uses current global temperature_K/reference_temperature_K and BOWING_PARAMETERS.
	"""
	if not (0.0 <= x_min <= 1.0 and 0.0 <= x_max <= 1.0):
		raise ValueError("x_min/x_max must be within [0,1]")
	if x_max < x_min:
		raise ValueError("x_max must be >= x_min")

	xs = _linspace(x_min, x_max, points)

	do_strained = include_strain or compare_strain_states
	if do_strained and substrate is None:
		raise ValueError("substrate must be provided when include_strain=True or compare_strain_states=True")
	if do_strained:
		assert substrate is not None
		if not (0.0 <= relaxation_degree <= 1.0):
			raise ValueError(
				f"relaxation_degree must be within [0,1]. Got: {relaxation_degree}"
			)

	Eg_ind_un: list[float] = []
	Eg_dir_un: list[float] = []
	Eg_ind_st: list[float] = []
	Eg_dir_st: list[float] = []
	strain_available = do_strained
	strain_error: str | None = None

	for x in xs:
		comp = {A: 1.0 - x, B: x}
		_, p = get_material(name, composition=comp)
		Eg_ind_un.append(float(p["Eg_ind"]))
		Eg_dir_un.append(float(p["Eg_dir"]))

		if strain_available:
			try:
				ind = strained_film_band_edges_pseudomorphic_001(
					substrate=substrate,
					thin_film=name,
					substrate_composition=substrate_composition,
					thin_film_composition=comp,
					band_edge_kind="ind",
					valence_band=strained_valence_band,
					relaxation_degree=relaxation_degree,
				)
				dir_ = strained_film_band_edges_pseudomorphic_001(
					substrate=substrate,
					thin_film=name,
					substrate_composition=substrate_composition,
					thin_film_composition=comp,
					band_edge_kind="dir",
					valence_band=strained_valence_band,
					relaxation_degree=relaxation_degree,
				)
				Eg_ind_st.append(float(ind["Eg_film"]))
				Eg_dir_st.append(float(dir_["Eg_film"]))
			except KeyError as exc:
				strain_available = False
				strain_error = str(exc)
				# Fill remainder with unstrained values so plotting stays aligned.
				Eg_ind_st.append(Eg_ind_un[-1])
				Eg_dir_st.append(Eg_dir_un[-1])
		else:
			# Keep lengths consistent even when strained calculation is unavailable.
			if do_strained:
				Eg_ind_st.append(Eg_ind_un[-1])
				Eg_dir_st.append(Eg_dir_un[-1])

	# Summary rows (for printing) use either strained or unstrained as the primary dataset.
	rows: list[tuple[float, float, float, float, str]] = []
	primary_ind = Eg_ind_st if (do_strained and include_strain and strain_available) else Eg_ind_un
	primary_dir = Eg_dir_st if (do_strained and include_strain and strain_available) else Eg_dir_un
	for x, ei, ed in zip(xs, primary_ind, primary_dir):
		delta = ed - ei
		kind = "dir" if ed < ei else "ind"
		rows.append((x, float(ei), float(ed), float(delta), kind))

	print(f"\nComposition sweep: {name} = {A}(1-x){B}(x)")
	print(f"  T={temperature_K:.1f} K (Varshni), T_ref={reference_temperature_K:.1f} K")
	if do_strained:
		if strain_available:
			print(
				f"  Strain curve: ON (001 on {substrate}, valence={strained_valence_band}, "
				f"relaxation={relaxation_degree:.3f})"
			)
		else:
			print(f"  Strain curve: requested but unavailable -> OFF (fallback). Reason: {strain_error}")
	if compare_strain_states:
		print("  Unstrained curve: ON (bulk alloy bandgaps)")
	print(f"  x range: {x_min:.4f} .. {x_max:.4f}  (points={points})")
	print("\n    x(B)     Eg_ind(eV)   Eg_dir(eV)   (Eg_dir-Eg_ind)    min-kind")
	print("  -------    ---------   ---------   --------------     --------")
	for x, Eg_ind, Eg_dir, delta, kind in rows:
		print(f"  {x:7.4f}    {Eg_ind:9.4f}   {Eg_dir:9.4f}   {delta:14.4f}     {kind}")

	# Estimate crossover where Eg_dir == Eg_ind (delta=0)
	x_cross: float | None = None
	y_cross: float | None = None
	for (x0, ei0, ed0, d0, _k0), (x1, ei1, ed1, d1, _k1) in zip(rows, rows[1:]):
		if d0 == 0.0:
			x_cross = x0
			y_cross = ei0
			break
		if (d0 < 0.0 and d1 > 0.0) or (d0 > 0.0 and d1 < 0.0):
			# linear interpolation in delta
			t = -d0 / (d1 - d0)
			x_cross = x0 + t * (x1 - x0)
			y_cross = ei0 + t * (ei1 - ei0)
			break
	if x_cross is not None:
		print(f"\nEstimated crossover (Eg_dir = Eg_ind): x ≈ {x_cross:.4f}")
	else:
		print("\nEstimated crossover: not found in this x-range")

	if not plot:
		return

	try:
		import matplotlib.pyplot as plt
	except ImportError:
		print("\n[plot] matplotlib is not installed. Install with: pip install matplotlib")
		return

	fig, ax = plt.subplots(figsize=(6.2, 4.0))
	xs_plot = list(xs)

	# Plot style: color = ind/dir, linestyle = unstrained/strained
	def _lighten_rgb(rgb: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
		# amount: 0 -> unchanged, 1 -> white
		return tuple(min(1.0, max(0.0, c + (1.0 - c) * amount)) for c in rgb)  # type: ignore[return-value]

	unstrained_ind_color = _lighten_rgb(nn_blue, 0.4)
	unstrained_dir_color = _lighten_rgb(violet, 0.4)
	ax.plot(
		xs_plot,
		Eg_ind_un,
		label="Eg_ind (L) unstrained",
		color=unstrained_ind_color,
		linewidth=4.0,
		linestyle=":",
	)
	ax.plot(
		xs_plot,
		Eg_dir_un,
		label="Eg_dir (Γ) unstrained",
		color=unstrained_dir_color,
		linewidth=4.0,
		linestyle=":",
	)
	if do_strained and strain_available:
		ax.plot(xs_plot, Eg_ind_st, label="Eg_ind (L) strained", color=nn_blue, linewidth=4.0, linestyle="-")
		ax.plot(xs_plot, Eg_dir_st, label="Eg_dir (Γ) strained", color=violet, linewidth=4.0, linestyle="-")
	elif do_strained:
		# Still show a legend entry indicating the requested strained curve fell back.
		# ax.plot(xs_plot, Eg_ind_st, label="Eg_ind (L) strained (fallback)", color=nn_blue, linewidth=1.6, linestyle="-")
		ax.plot(xs_plot, Eg_dir_st, label="Eg_dir (Γ) strained (fallback)", color=violet, linewidth=4.0, linestyle="-")
	ax.tick_params(which="both", direction="in")
	ax.xaxis.set_tick_params(direction='in', labelsize = 16, width = 4.0, length = 9.0, pad=7.5)
	ax.yaxis.set_tick_params(direction='in', labelsize = 16, width = 4.0, length = 9.0)
	ax.set_xlabel(f"x in {A}(1-x){B}(x)", fontsize=16)
	ax.set_ylabel("Bandgap (eV)", fontsize=16)
	ax.grid(True, alpha=0.25)
	for spine in ax.spines.values():
		spine.set_linewidth(4.0)

	# Optional marker at a specific composition (e.g. x=0.08 = 8%).
	if mark_x is not None and (x_min <= mark_x <= x_max):
		def _interp_linear(xs_: list[float], ys_: list[float], x_: float) -> float:
			if x_ <= xs_[0]:
				return ys_[0]
			if x_ >= xs_[-1]:
				return ys_[-1]
			for i in range(len(xs_) - 1):
				x0, x1 = xs_[i], xs_[i + 1]
				if x0 <= x_ <= x1:
					t = 0.0 if x1 == x0 else (x_ - x0) / (x1 - x0)
					return ys_[i] + t * (ys_[i + 1] - ys_[i])
			return ys_[-1]

		# Use strained values if available; otherwise unstrained.
		Eg_ind_for_mark = Eg_ind_st if (do_strained and strain_available) else Eg_ind_un
		Eg_dir_for_mark = Eg_dir_st if (do_strained and strain_available) else Eg_dir_un
		y_ind_m = _interp_linear(xs_plot, Eg_ind_for_mark, mark_x)
		y_dir_m = _interp_linear(xs_plot, Eg_dir_for_mark, mark_x)
		# ax.axvline(mark_x, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
		# ax.plot([mark_x], [y_ind_m], marker="o", color=nn_blue, markersize=4)
		# ax.plot([mark_x], [y_dir_m], marker="o", color=violet, markersize=4)
		y_all = Eg_ind_un + Eg_dir_un
		if do_strained:
			y_all = y_all + Eg_ind_st + Eg_dir_st
		y_span = (max(y_all) - min(y_all)) if len(y_all) > 1 else 1.0
		dx = 0.03 * (x_max - x_min) if (x_max > x_min) else 0.02
		label = (
			f"x({B}) = {mark_x:.3f} ({mark_x*100:.1f}%)\n"
			f"Eg_ind = {y_ind_m:.3f} eV\n"
			f"Eg_dir = {y_dir_m:.3f} eV"
		)
		# ax.annotate(
		# 	label,
		# 	xy=(mark_x, max(y_ind_m, y_dir_m)),
		# 	xytext=(min(mark_x + dx, x_max), max(y_ind_m, y_dir_m) + 0.08 * y_span),
		# 	textcoords="data",
		# 	ha="left",
		# 	va="bottom",
		# 	arrowprops={"arrowstyle": "-", "color": "black", "alpha": 0.6},
		# )

	# # Annotate crossover composition on the plot
	# if (x_cross is not None) and (y_cross is not None):
	# 	ax.plot([x_cross], [y_cross], marker="o", color="black", markersize=4)
	# 	x_label = f"x({B}) ≈ {x_cross:.4f} ({x_cross*100:.2f}%)"
	# 	ax.annotate(
	# 		x_label,
	# 		xy=(x_cross, y_cross),
	# 		xytext=(min(max(x_cross + 0.03, x_min), x_max), y_cross + 0.15),
	# 		textcoords="data",
	# 		ha="left",
	# 		va="bottom",
	# 		arrowprops={"arrowstyle": "-", "color": "black", "alpha": 0.7},
	# 	)

	# ax.legend()
	fig.tight_layout()

	if save_path is not None:
		saved = _save_figure(fig, save_path, also_save_svg=True, dpi=200)
		print("\nSaved sweep plot:")
		for p in saved:
			print(f"  {p}")

	if show:
		plt.show()
	else:
		plt.close(fig)


def _composition_label(composition: dict[str, float]) -> str:
	comp = _normalize_composition(composition)
	# stable order: Si, Ge, Sn first if present, then others alphabetically
	priority = {"Si": 0, "Ge": 1, "Sn": 2}
	items = sorted(comp.items(), key=lambda kv: (priority.get(kv[0], 99), kv[0]))
	return "".join(f"{k}{v:.2f}" for k, v in items)


def get_material(name: str, *, composition: dict[str, float] | None = None) -> tuple[str, dict[str, float]]:
	"""Return (label, params) for an element/compound or an alloy.

	- If composition is None: `name` must exist in MATERIALS.
	- If composition is provided: parameters are linearly interpolated and label becomes
	  f"{name}({_composition_label(composition)})".
	"""
	if composition is not None:
		params = make_alloy_params(composition)
		params = apply_temperature_to_params(
			params,
			temperature_K=temperature_K,
			reference_temperature_K=reference_temperature_K,
		)
		return (f"{name}({_composition_label(composition)})", params)

	try:
		params = MATERIALS[name]
		params = apply_temperature_to_params(
			params,
			temperature_K=temperature_K,
			reference_temperature_K=reference_temperature_K,
		)
		return (name, params)
	except KeyError as exc:
		raise KeyError(f"Unknown material: {name}. Available: {list(MATERIALS)}") from exc


def band_offsets_unstrained(
	substrate: str,
	thin_film: str,
	*,
	substrate_composition: dict[str, float] | None = None,
	thin_film_composition: dict[str, float] | None = None,
	band_edge_kind: str = "ind",
) -> dict[str, float]:
	"""Band offsets film - substrate (unstrained), using absolute band edges.

	Returns
	-------
	dict with keys:
	  - dEv: Ev(film) - Ev(substrate) [eV]
	  - dEc: Ec(film) - Ec(substrate) [eV]
	  - dEg: Eg(film) - Eg(substrate) [eV]
	"""
	_, sub = get_material(substrate, composition=substrate_composition)
	_, film = get_material(thin_film, composition=thin_film_composition)

	kind = band_edge_kind.lower().strip()
	if kind not in {"ind", "dir", "both"}:
		raise ValueError(
			f"band_edge_kind must be 'ind', 'dir', or 'both'. Got: {band_edge_kind}"
		)

	def vbm(p: dict[str, float]) -> float:
		return p["Ev_av"] + p["delta0"] / 3.0

	def band_edges(p: dict[str, float], edge_kind: str) -> tuple[float, float, float]:
		Ev = vbm(p)
		Eg = p[f"Eg_{edge_kind}"]
		Ec = p[f"Ec_{edge_kind}"]
		return Ev, Ec, Eg

	# Valence offsets are independent of ind/dir choice in this simplified model
	Ev_sub = vbm(sub)
	Ev_film = vbm(film)
	dEv_av = film["Ev_av"] - sub["Ev_av"]
	dEv = Ev_film - Ev_sub

	if kind in {"ind", "dir"}:
		_, Ec_sub, Eg_sub = band_edges(sub, kind)
		_, Ec_film, Eg_film = band_edges(film, kind)
		dEc = Ec_film - Ec_sub
		dEg = Eg_film - Eg_sub
		return {"dEv_av": dEv_av, "dEv": dEv, "dEc": dEc, "dEg": dEg, "kind": kind}

	# both
	_, Ec_sub_ind, Eg_sub_ind = band_edges(sub, "ind")
	_, Ec_film_ind, Eg_film_ind = band_edges(film, "ind")
	_, Ec_sub_dir, Eg_sub_dir = band_edges(sub, "dir")
	_, Ec_film_dir, Eg_film_dir = band_edges(film, "dir")

	return {
		"dEv_av": dEv_av,
		"dEv": dEv,
		"dEc_ind": Ec_film_ind - Ec_sub_ind,
		"dEg_ind": Eg_film_ind - Eg_sub_ind,
		"dEc_dir": Ec_film_dir - Ec_sub_dir,
		"dEg_dir": Eg_film_dir - Eg_sub_dir,
		"kind": "both",
	}



def pseudomorphic_strain(
	substrate: str,
	thin_film: str,
	*,
	substrate_composition: dict[str, float] | None = None,
	thin_film_composition: dict[str, float] | None = None,
	relaxation_degree: float = 0.0,
) -> dict[str, float]:
	"""Compute geometric strain components for a cubic film on a cubic substrate.

	This is *only* the strain tensor components (no band-edge shifts).

	Parameters
	----------
	relaxation_degree:
		0.0 for fully pseudomorphic (coherent) film; 1.0 for fully relaxed film.
		Intermediate values approximate partial relaxation by reducing the in-plane strain:
		  eps_parallel_effective = (1 - relaxation_degree) * eps_parallel_pseudomorphic
	"""
	if not (0.0 <= relaxation_degree <= 1.0):
		raise ValueError(
			f"relaxation_degree must be within [0,1]. Got: {relaxation_degree}"
		)
	_, sub = get_material(substrate, composition=substrate_composition)
	_, film = get_material(thin_film, composition=thin_film_composition)

	eps_parallel_pseudo = (sub["a"] - film["a"]) / film["a"]
	eps_parallel = (1.0 - relaxation_degree) * eps_parallel_pseudo
	eps_perp = -2.0 * (film["c12"] / film["c11"]) * eps_parallel
	eps_hydro = 2.0 * eps_parallel + eps_perp
	return {
		"relaxation_degree": relaxation_degree,
		"eps_parallel_pseudomorphic": eps_parallel_pseudo,
		"eps_parallel": eps_parallel,
		"eps_perp": eps_perp,
		"eps_hydro": eps_hydro,
	}


def _require_keys(p: dict[str, float], keys: set[str], *, context: str) -> None:
	missing = [k for k in sorted(keys) if k not in p]
	if missing:
		raise KeyError(
			f"Missing required parameters for {context}: {missing}. "
			"Add them to MATERIALS (and to all base materials if you want alloy interpolation)."
		)


def _require_any_keys(
	p: dict[str, float],
	requirements: dict[str, tuple[str, ...]],
	*,
	context: str,
) -> None:
	"""Require that for each conceptual parameter, at least one alias key exists."""
	missing = [
		concept for concept, aliases in requirements.items() if not any(a in p for a in aliases)
	]
	if missing:
		details = {concept: list(requirements[concept]) for concept in missing}
		raise KeyError(
			f"Missing required parameters for {context}: {details}. "
			"Add them to MATERIALS (and to all base materials if you want alloy interpolation)."
		)


def _get_any_key(p: dict[str, float], aliases: tuple[str, ...], *, context: str) -> float:
	for a in aliases:
		if a in p:
			return p[a]
	raise KeyError(
		f"Missing required parameters for {context}: expected one of {list(aliases)}. "
		"Add it to MATERIALS (and to all base materials if you want alloy interpolation)."
	)


def _ge_sn_paper_deformation_potentials(
	thin_film: str,
	thin_film_composition: dict[str, float] | None,
) -> tuple[float, float, float]:
	"""Return (b, a_c_ind, a_c_dir) from the optional GeSn model.

	The paper model is defined for Ge_(1-y)Sn_y and uses a constant valence-band
	shear deformation potential.  The ``a_v`` value is retained from MATERIALS.
	"""
	if thin_film_composition is None:
		if thin_film == "Ge":
			y_sn = 0.0
		elif thin_film == "Sn":
			y_sn = 1.0
		else:
			raise ValueError(
				"The paper GeSn deformation-potential model requires "
				"thin_film_composition for an alloy."
			)
	else:
		comp = _normalize_composition(thin_film_composition)
		unknown = set(comp) - {"Ge", "Sn"}
		if unknown:
			raise ValueError(
				"The paper deformation-potential model is defined only for Ge_(1-y)Sn_y. "
				f"Unsupported composition components: {sorted(unknown)}"
			)
		y_sn = comp.get("Sn", 0.0)

	a_c_dir = -9.46 + 2.33 * y_sn
	a_c_ind = -3.60 + 1.77 * y_sn
	b = -1.88
	return b, a_c_ind, a_c_dir


def strained_film_band_edges_pseudomorphic_001(
	substrate: str,
	thin_film: str,
	*,
	substrate_composition: dict[str, float] | None = None,
	thin_film_composition: dict[str, float] | None = None,
	band_edge_kind: str = "ind",
	valence_band: str = "vbm",
	relaxation_degree: float = 0.0,
) -> dict[str, float]:
	"""Band edges for a (001) film including simple deformation-potential shifts.

	Model
	-----
	For cubic semiconductors with biaxial strain (001):
	- Hydrostatic strain: Tr(eps) = eps_hydro
	- Average valence shift: A = Ev_av + a_v * Tr(eps)
	- Valence shear term: Q = b * (eps_parallel - eps_perp)
	- Unstrained (k=0): HH=LH=Ev_av + Δ0/3, SO=Ev_av - 2Δ0/3
	- Strained (k=0):
	  HH is decoupled: Ev(HH) = A + Δ0/3 - Q
	  LH and SO mix (2×2 diagonalization with coupling √2·Q)
	- Conduction (simple hydrostatic-only): Ec = Ec0 + a_c * Tr(eps)

	If ``use_paper_deformation_potentials`` is enabled, the Ge_(1-y)Sn_y
	composition-dependent values are used for ``a_c_ind``, ``a_c_dir``, and
	``b`` instead of the interpolated MATERIALS values.

	Required film parameters (keys in MATERIALS)
	-------------------------------------------
	- a_v (or av): valence hydrostatic deformation potential [eV]
	- b: valence shear deformation potential [eV]
	- a_c_ind (or ac_ind): conduction hydrostatic deformation potential for indirect edge [eV]
	- a_c_dir (or ac_dir): conduction hydrostatic deformation potential for direct edge [eV]

	Notes
	-----
	This is still simplified on the conduction side (no valley splitting for X/L). On the
	valence side it includes LH–SO mixing due to strain.
	"""
	_, sub = get_material(substrate, composition=substrate_composition)
	_, film = get_material(thin_film, composition=thin_film_composition)

	kind = band_edge_kind.lower().strip()
	if kind not in {"ind", "dir"}:
		raise ValueError(f"band_edge_kind must be 'ind' or 'dir'. Got: {band_edge_kind}")

	strain = pseudomorphic_strain(
		substrate=substrate,
		thin_film=thin_film,
		substrate_composition=substrate_composition,
		thin_film_composition=thin_film_composition,
		relaxation_degree=relaxation_degree,
	)

	_required_any = {
		"a_v": ("a_v", "av"),
		"b": ("b",),
		"a_c_ind": ("a_c_ind", "ac_ind"),
		"a_c_dir": ("a_c_dir", "ac_dir"),
	}
	_require_any_keys(film, _required_any, context=f"strain band shifts (film={thin_film})")

	Delta0 = film["delta0"]
	a_v = _get_any_key(film, ("a_v", "av"), context=f"strain band shifts (film={thin_film})")
	b = _get_any_key(film, ("b",), context=f"strain band shifts (film={thin_film})")
	a_c_ind = _get_any_key(
		film, ("a_c_ind", "ac_ind"), context=f"strain band shifts (film={thin_film})"
	)
	a_c_dir = _get_any_key(
		film, ("a_c_dir", "ac_dir"), context=f"strain band shifts (film={thin_film})"
	)
	if use_paper_deformation_potentials:
		b, a_c_ind, a_c_dir = _ge_sn_paper_deformation_potentials(
			thin_film, thin_film_composition
		)
	A = film["Ev_av"] + a_v * strain["eps_hydro"]
	Q = b * (strain["eps_perp"] - strain["eps_parallel"])

	# HH (decoupled)
	Ev_hh = A + Delta0 / 3.0 - Q

	# LH / SO mixing via 2x2 block:
	#   [ A + Δ0/3 + Q     √2 Q ]
	#   [ √2 Q             A - 2Δ0/3 ]
	# Eigenvalues:
	#   Ev = A - Δ0/6 + Q/2 ± 0.5*sqrt(Δ0^2 + 2Δ0 Q + 9 Q^2)
	root = (Delta0 * Delta0) + (2.0 * Delta0 * Q) + (9.0 * Q * Q)
	# Numerical guard: root should be >=0 but allow tiny negatives from float roundoff.
	if root < 0 and root > -1e-15:
		root = 0.0
	from math import sqrt
	D = sqrt(root)
	Ev_lh = A - Delta0 / 6.0 + Q / 2.0 + 0.5 * D
	Ev_so = A - Delta0 / 6.0 + Q / 2.0 - 0.5 * D

	vb = valence_band.lower().strip()
	if vb not in {"vbm", "hh", "lh", "so"}:
		raise ValueError(f"valence_band must be 'vbm', 'hh', 'lh', or 'so'. Got: {valence_band}")
	if vb == "hh":
		Ev = Ev_hh
	elif vb == "lh":
		Ev = Ev_lh
	elif vb == "so":
		Ev = Ev_so
	else:
		Ev = max(Ev_hh, Ev_lh)

	Ec0 = film[f"Ec_{kind}"]
	if kind == "ind":
		a_c = a_c_ind
	else:
		a_c = a_c_dir
	Ec = Ec0 + a_c * strain["eps_hydro"]

	Ec_sub = sub[f"Ec_{kind}"]
	Ev_sub = sub["Ev_av"] + sub["delta0"] / 3.0
	Eg_sub = Ec_sub - Ev_sub
	Eg = Ec - Ev

	return {
		"kind": kind,
		"valence_band": vb,
		"relaxation_degree": relaxation_degree,
		"Ev_sub": Ev_sub,
		"Ec_sub": Ec_sub,
		"Eg_sub": Eg_sub,
		"Ev_film": Ev,
		"Ev_film_hh": Ev_hh,
		"Ev_film_lh": Ev_lh,
		"Ev_film_so": Ev_so,
		"Ec_film": Ec,
		"Eg_film": Eg,
		"dEv": Ev - Ev_sub,
		"dEc": Ec - Ec_sub,
		"dEg": Eg - Eg_sub,
		"eps_parallel_pseudomorphic": strain["eps_parallel_pseudomorphic"],
		"eps_parallel": strain["eps_parallel"],
		"eps_perp": strain["eps_perp"],
		"eps_hydro": strain["eps_hydro"],
	}

	


def plot_band_alignment(
	substrate: str,
	thin_film: str,
	*,
	substrate_composition: dict[str, float] | None = None,
	thin_film_composition: dict[str, float] | None = None,
	band_edge_kind: str = "ind",
	include_strain: bool = False,
	strained_valence_band: str = "vbm",
	relaxation_degree: float = 0.0,
	plot_hh_lh: bool = True,
	plot_so: bool = False,
	save_path: str | Path | None = None,
	show: bool = True,
) -> Path | None:
	"""Plot Ev/Ec lineup for substrate and thin film.

	Assumes Ev_av and Ec_ind are absolute energies (eV) relative to the same reference.
	"""
	try:
		import matplotlib.pyplot as plt
	except ImportError:
		print("matplotlib is not installed. Install with: pip install matplotlib")
		return None

	sub_label, sub = get_material(substrate, composition=substrate_composition)
	film_label, film = get_material(thin_film, composition=thin_film_composition)
	kind = band_edge_kind.lower().strip()
	if kind not in {"ind", "dir", "both"}:
		raise ValueError(
			f"band_edge_kind must be 'ind', 'dir', or 'both'. Got: {band_edge_kind}"
		)

	# x positions for two materials
	x_left, x_if, x_right = 0.0, 0.5, 1.0

	Ev_sub = sub["Ev_av"] + sub["delta0"] / 3.0
	Ev_film = film["Ev_av"] + film["delta0"] / 3.0

	Ec_sub_ind = sub["Ec_ind"]
	Ec_film_ind = film["Ec_ind"]
	Ec_sub_dir = sub["Ec_dir"]
	Ec_film_dir = film["Ec_dir"]

	Ev_film_hh: float | None = None
	Ev_film_lh: float | None = None
	Ev_film_so: float | None = None

	if include_strain:
		# Apply strain to film only (substrate assumed relaxed)
		try:
			ind = strained_film_band_edges_pseudomorphic_001(
				substrate=substrate,
				thin_film=thin_film,
				substrate_composition=substrate_composition,
				thin_film_composition=thin_film_composition,
				band_edge_kind="ind",
				valence_band=strained_valence_band,
				relaxation_degree=relaxation_degree,
			)
			dir_ = strained_film_band_edges_pseudomorphic_001(
				substrate=substrate,
				thin_film=thin_film,
				substrate_composition=substrate_composition,
				thin_film_composition=thin_film_composition,
				band_edge_kind="dir",
				valence_band=strained_valence_band,
				relaxation_degree=relaxation_degree,
			)
			# Valence is shared; pick from indirect calculation
			Ev_film = ind["Ev_film"]
			Ev_film_hh = ind.get("Ev_film_hh")
			Ev_film_lh = ind.get("Ev_film_lh")
			Ev_film_so = ind.get("Ev_film_so")
			Ec_film_ind = ind["Ec_film"]
			Ec_film_dir = dir_["Ec_film"]
		except KeyError as exc:
			print(f"[plot] Strain shifts skipped: {exc}")

	if kind == "ind":
		ec_pairs = [(Ec_sub_ind, Ec_film_ind)]
	elif kind == "dir":
		ec_pairs = [(Ec_sub_dir, Ec_film_dir)]
	else:
		ec_pairs = [(Ec_sub_ind, Ec_film_ind), (Ec_sub_dir, Ec_film_dir)]

	all_energies = [Ev_sub, Ev_film]
	if plot_hh_lh and include_strain and (Ev_film_hh is not None) and (Ev_film_lh is not None):
		all_energies.extend([Ev_film_hh, Ev_film_lh])
	if plot_so and include_strain and (Ev_film_so is not None):
		all_energies.append(Ev_film_so)
	for ec_sub, ec_film in ec_pairs:
		all_energies.extend([ec_sub, ec_film])

	y_min = min(all_energies) - 0.4
	y_max = max(all_energies) + 0.4

	fig, ax = plt.subplots(figsize=(6.2, 4.0))

	# valence band (same for ind/dir)
	ax.hlines(Ev_sub, x_left, x_if, linewidth=3, color="black")
	if plot_hh_lh and include_strain and (Ev_film_hh is not None) and (Ev_film_lh is not None):
		# Draw both HH/LH on film side; keep Ev_film as the chosen VBM for annotations.
		ax.hlines(Ev_film_hh, x_if, x_right, linewidth=2.5, color="black")
		ax.hlines(Ev_film_lh, x_if, x_right, linewidth=2.5, color="black", linestyle="--")
		if plot_so and (Ev_film_so is not None):
			ax.hlines(Ev_film_so, x_if, x_right, linewidth=4.0, color="black", linestyle=":")
	else:
		ax.hlines(Ev_film, x_if, x_right, linewidth=3, color="black")

	# conduction bands
	if kind in {"ind", "both"}:
		ax.hlines(Ec_sub_ind, x_left, x_if, linewidth=3, color=nn_blue)
		ax.hlines(Ec_film_ind, x_if, x_right, linewidth=3, color=nn_blue)
	if kind in {"dir", "both"}:
		ax.hlines(Ec_sub_dir, x_left, x_if, linewidth=3, color=violet)
		ax.hlines(Ec_film_dir, x_if, x_right, linewidth=3, color=violet)

	# interface marker
	ax.axvline(x_if, color="black", linewidth=1, alpha=0.6)

	# labels near each region
	ax.text(x_left + 0.02, Ev_sub, "Ev", va="top", ha="left", color="black")
	if plot_hh_lh and include_strain and (Ev_film_hh is not None) and (Ev_film_lh is not None):
		ax.text(x_right - 0.02, Ev_film_hh, "Ev(HH)", va="top", ha="right", color="black")
		ax.text(x_right - 0.02, Ev_film_lh, "Ev(LH)", va="top", ha="right", color="black")
		if plot_so and (Ev_film_so is not None):
			ax.text(x_right - 0.02, Ev_film_so, "Ev(SO)", va="top", ha="right", color="black")
	else:
		ax.text(x_right - 0.02, Ev_film, "Ev", va="top", ha="right", color="black")
	if kind in {"ind", "both"}:
		ax.text(x_left + 0.02, Ec_sub_ind, "Ec(ind)", va="bottom", ha="left", color=nn_blue)
		ax.text(x_right - 0.02, Ec_film_ind, "Ec(ind)", va="bottom", ha="right", color=nn_blue)
	if kind in {"dir", "both"}:
		ax.text(x_left + 0.02, Ec_sub_dir, "Ec(dir)", va="bottom", ha="left", color=violet)
		ax.text(x_right - 0.02, Ec_film_dir, "Ec(dir)", va="bottom", ha="right", color=violet)

	# offsets annotation (film - substrate)
	dEv = Ev_film - Ev_sub
	ax.annotate(
		f"ΔEv = {dEv:+.3f} eV",
		xy=(x_if, (Ev_sub + Ev_film) / 2.0),
		xytext=(x_if, (Ev_sub + Ev_film) / 2.0 + 0.25),
		ha="center",
		arrowprops={"arrowstyle": "-"},
	)
	if kind in {"ind", "both"}:
		dEc_ind = Ec_film_ind - Ec_sub_ind
		ax.annotate(
			f"ΔEc(ind) = {dEc_ind:+.3f} eV",
			xy=(x_if, (Ec_sub_ind + Ec_film_ind) / 2.0),
			xytext=(x_if, (Ec_sub_ind + Ec_film_ind) / 2.0 + 0.25),
			ha="center",
			color=nn_blue,
			arrowprops={"arrowstyle": "-", "color": nn_blue},
		)
	if kind in {"dir", "both"}:
		dEc_dir = Ec_film_dir - Ec_sub_dir
		ax.annotate(
			f"ΔEc(dir) = {dEc_dir:+.3f} eV",
			xy=(x_if, (Ec_sub_dir + Ec_film_dir) / 2.0),
			xytext=(x_if, (Ec_sub_dir + Ec_film_dir) / 2.0 + 0.25),
			ha="center",
			color=violet,
			arrowprops={"arrowstyle": "-", "color": violet},
		)

	ax.set_xlim(x_left, x_right)
	ax.set_ylim(y_min, y_max)
	ax.set_xticks([0.25, 0.75], [sub_label, film_label])
	ax.set_ylabel("Energy (eV, common reference)")
	plot_kind_title = "ind" if kind == "ind" else "dir" if kind == "dir" else "ind+dir"
	ax.set_title(f"Band alignment ({plot_kind_title}): {sub_label} / {film_label}")
	ax.grid(True, axis="y", alpha=0.25)
	fig.tight_layout()

	out_path: Path | None = None
	if save_path is not None:
		out_path = Path(save_path)
		saved = _save_figure(fig, out_path, also_save_svg=True, dpi=200)
		# Keep return value as the primary requested path.
		# (Additional SVG is saved automatically when the primary is a PDF.)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return out_path


def _format_ev(x: float) -> str:
	return f"{x:+.3f} eV"


def _sanitize_filename_component(text: str) -> str:
	# Keep it simple/portable
	return (
		text.replace("(", "")
		.replace(")", "")
		.replace("=", "")
		.replace(" ", "")
		.replace(".", "p")
	)


def main() -> None:
	if do_composition_sweep:
		A, B = sweep_endpoints
		mark_x: float | None = None
		if sweep_mark_x_percent is not None:
			mark_x = sweep_mark_x_percent / 100.0
		save_path: Path | None = None
		if sweep_plot and sweep_save_plot:
			if sweep_plot_save_path is not None:
				save_path = Path(sweep_plot_save_path)
			else:
				if sweep_compare_strain_states:
					strain_tag = "compare"
					relax_tag = f"R{relaxation_degree:.3f}"  # strained curve uses this
				else:
					strain_tag = "strained" if include_strain_band_shifts else "unstrained"
					relax_tag = f"R{relaxation_degree:.3f}" if include_strain_band_shifts else ""
				tag = (
					f"{sweep_alloy_name}_{strain_tag}{relax_tag}_{A}1m{B}x_T{temperature_K:.0f}K_"
					f"x{sweep_x_min:.3f}-{sweep_x_max:.3f}_n{sweep_points}"
				)
				save_path = Path(__file__).with_name(f"bandgaps_vs_composition_{_sanitize_filename_component(tag)}.pdf")

		if sweep_compare_strain_states and include_strain_band_shifts and relaxation_degree >= 0.999:
			print(
				"\n[sweep][Warning] relaxation_degree is ~1.0 (fully relaxed). "
				"The 'strained' curve has ~zero strain and will overlap the unstrained curve. "
				"Set relaxation_degree closer to 0 for pseudomorphic strain, or an intermediate value for partial relaxation."
			)
		sweep_binary_alloy_bandgaps(
			A,
			B,
			name=sweep_alloy_name,
			x_min=sweep_x_min,
			x_max=sweep_x_max,
			points=sweep_points,
			plot=sweep_plot,
 			mark_x=mark_x,
			include_strain=include_strain_band_shifts,
			compare_strain_states=sweep_compare_strain_states,
			relaxation_degree=relaxation_degree,
			substrate=substrate_name,
			substrate_composition=substrate_composition,
			strained_valence_band=strained_valence_band,
			save_path=save_path,
			show=True,
		)
		return

	offsets = band_offsets_unstrained(
		substrate=substrate_name,
		thin_film=thin_film_name,
		substrate_composition=substrate_composition,
		thin_film_composition=thin_film_composition,
		band_edge_kind=band_edge_kind,
	)
	strain = pseudomorphic_strain(
		substrate=substrate_name,
		thin_film=thin_film_name,
		substrate_composition=substrate_composition,
		thin_film_composition=thin_film_composition,
		relaxation_degree=relaxation_degree,
	)
	sub_label, _sub = get_material(substrate_name, composition=substrate_composition)
	film_label, _film = get_material(thin_film_name, composition=thin_film_composition)

	if substrate_composition is not None:
		print(f"Substrate composition (raw): {substrate_composition}")
		print(f"Substrate composition (normalized): {_normalize_composition(substrate_composition)}")
	if thin_film_composition is not None:
		print(f"Thin film composition (raw): {thin_film_composition}")
		print(f"Thin film composition (normalized): {_normalize_composition(thin_film_composition)}")

	print(f"Substrate: {sub_label}")
	print(f"Thin film: {film_label}")
	print(
		f"\nTemperature for bandgaps (Varshni): T={temperature_K:.1f} K, "
		f"T_ref={reference_temperature_K:.1f} K"
	)
	print(
		f"  Substrate Eg_ind={_sub['Eg_ind']:.3f} eV, Eg_dir={_sub['Eg_dir']:.3f} eV"
	)
	print(
		f"  Thin film  Eg_ind={_film['Eg_ind']:.3f} eV, Eg_dir={_film['Eg_dir']:.3f} eV"
	)
	print(f"\nUnstrained band offsets (film - substrate), kind={offsets['kind']}:")
	print(f"  ΔEv_av = {offsets['dEv_av']:+.3f} eV")
	print(f"  ΔEv    = {offsets['dEv']:+.3f} eV")
	if offsets["kind"] in {"ind", "dir"}:
		print(f"  ΔEc = {offsets['dEc']:+.3f} eV")
		print(f"  ΔEg = {offsets['dEg']:+.3f} eV")
	else:
		print(f"  ΔEc(ind) = {offsets['dEc_ind']:+.3f} eV")
		print(f"  ΔEg(ind) = {offsets['dEg_ind']:+.3f} eV")
		print(f"  ΔEc(dir) = {offsets['dEc_dir']:+.3f} eV")
		print(f"  ΔEg(dir) = {offsets['dEg_dir']:+.3f} eV")

	print("\nPseudomorphic strain in film (geometry only):")
	print(f"  relaxation = {relaxation_degree:.3f}  (0=pseudomorphic, 1=relaxed)")
	print(f"  ε_parallel(pseudo) = {strain['eps_parallel_pseudomorphic']:+.5f}")
	print(f"  ε_parallel = {strain['eps_parallel']:+.5f}")
	print(f"  ε_perp     = {strain['eps_perp']:+.5f}")
	print(f"  ε_hydro    = {strain['eps_hydro']:+.5f}")

	if include_strain_band_shifts:
		print("\nStrain-induced band-edge shifts ((001) film, with relaxation):")
		try:
			if band_edge_kind.lower().strip() in {"ind", "dir"}:
				strained = strained_film_band_edges_pseudomorphic_001(
					substrate=substrate_name,
					thin_film=thin_film_name,
					substrate_composition=substrate_composition,
					thin_film_composition=thin_film_composition,
					band_edge_kind=band_edge_kind,
					valence_band=strained_valence_band,
					relaxation_degree=relaxation_degree,
				)
				print(f"  kind={strained['kind']}, valence={strained['valence_band']}")
				print(f"  ΔEv(strain) = {strained['dEv']:+.3f} eV")
				print(f"  ΔEc(strain) = {strained['dEc']:+.3f} eV")
				print(f"  ΔEg(strain) = {strained['dEg']:+.3f} eV")
				print(f"  Eg_film(strain) = {strained['Eg_film']:.3f} eV")
				print(f"  Film Ev(HH) = {strained['Ev_film_hh']:+.3f} eV")
				print(f"  Film Ev(LH) = {strained['Ev_film_lh']:+.3f} eV")
				print(f"  Film Ev(SO) = {strained['Ev_film_so']:+.3f} eV")
			else:
				ind = strained_film_band_edges_pseudomorphic_001(
					substrate=substrate_name,
					thin_film=thin_film_name,
					substrate_composition=substrate_composition,
					thin_film_composition=thin_film_composition,
					band_edge_kind="ind",
					valence_band=strained_valence_band,
					relaxation_degree=relaxation_degree,
				)
				dir_ = strained_film_band_edges_pseudomorphic_001(
					substrate=substrate_name,
					thin_film=thin_film_name,
					substrate_composition=substrate_composition,
					thin_film_composition=thin_film_composition,
					band_edge_kind="dir",
					valence_band=strained_valence_band,
					relaxation_degree=relaxation_degree,
				)
				print(f"  valence={ind['valence_band']}")
				print(f"  ΔEv(strain)     = {ind['dEv']:+.3f} eV")
				print(f"  ΔEc_ind(strain) = {ind['dEc']:+.3f} eV")
				print(f"  ΔEg_ind(strain) = {ind['dEg']:+.3f} eV")
				print(f"  Eg_ind_film(strain) = {ind['Eg_film']:.3f} eV")
				print(f"  ΔEc_dir(strain) = {dir_['dEc']:+.3f} eV")
				print(f"  ΔEg_dir(strain) = {dir_['dEg']:+.3f} eV")
				print(f"  Eg_dir_film(strain) = {dir_['Eg_film']:.3f} eV")
				print(f"  Film Ev(HH)     = {ind['Ev_film_hh']:+.3f} eV")
				print(f"  Film Ev(LH)     = {ind['Ev_film_lh']:+.3f} eV")
				print(f"  Film Ev(SO)     = {ind['Ev_film_so']:+.3f} eV")
		except KeyError as exc:
			print(f"  (skipped) {exc}")
			print(
				"  Required film keys: av, b, ac_ind, ac_dir (eV). "
				"Add them to MATERIALS for all base materials involved (Si/Ge/Sn ...) so alloys can interpolate them."
			)
	else:
		print("\nTo include strain-induced band-edge shifts, set include_strain_band_shifts=True and add deformation potentials.")

	# Plot (and save) band alignment diagram
	strain_tag = "strained" if include_strain_band_shifts else "unstrained"
	relax_tag = f"_R{relaxation_degree:.3f}" if include_strain_band_shifts else ""
	out = plot_band_alignment(
		substrate=substrate_name,
		thin_film=thin_film_name,
		substrate_composition=substrate_composition,
		thin_film_composition=thin_film_composition,
		band_edge_kind=band_edge_kind,
		include_strain=include_strain_band_shifts,
		strained_valence_band=strained_valence_band,
		relaxation_degree=relaxation_degree,
		save_path=Path(__file__).with_name(
			f"band_alignment_{strain_tag}{relax_tag}_{band_edge_kind}_{_sanitize_filename_component(sub_label)}_{_sanitize_filename_component(film_label)}.pdf"
		),
		show=True,
	)
	if out is not None:
		print(f"\nSaved plot: {out}")
		if out.suffix.lower() == ".pdf":
			print(f"Saved plot: {out.with_suffix('.svg')}")


if __name__ == "__main__":
	main()

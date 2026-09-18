"""Repackage verified Chapter 5 results with revised prose and colour legends.

Reads frozen outputs only. Does not run a solver or modify an earlier study.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, SymLogNorm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/chapter5_measured_revision'
OUT = ROOT / 'output/chapter5_plain_revision'
TEX = OUT / 'overleaf'
FIG = TEX / 'ch5'
FIG.mkdir(parents=True, exist_ok=True)
source = (SOURCE / 'overleaf/chapter5.tex').read_text()
shutil.copytree(SOURCE / 'overleaf/ch5', FIG, dirs_exist_ok=True)

def block(label):
    for match in re.finditer(r'\\begin\{(table|figure|equation)\}.*?\\end\{\1\}', source, re.S):
        if '\\label{' + label + '}' in match.group():
            return match.group()
    raise ValueError(label)

def expand(text):
    return re.sub(r'@@([^@]+)@@', lambda m: block(m[1]), text)

chapter = r'''%============================================================
% CHAPTER 5: NUMERICAL STUDIES
%============================================================
\chapter{Numerical Studies}
\label{chap:numerical_studies}

%============================================================
% 5.1 STUDY OBJECTIVES AND SCOPE
%============================================================
\section{Study Objectives and Scope}
\label{sec:numerical_scope}
This chapter evaluates the original tube configuration and a proposed revision within the measured limits of the CTR platform. The study examines the available workspace, positional dexterity and numerical target reaching. It then compares path-following performance to check whether the proposed tubes also improve movement along a trajectory.

The original tubes remain the physical reference. Their earlier characterisation used deployment bounds based on tube lengths. The current comparison applies the measured carriage travel and actuator clearances to both configurations. It replaces the earlier estimated-hardware comparison that used a 35~mm grip offset. Improvements are calculated between designs under the same current hardware assumptions, rather than between studies with different movement limits.

The selected tubes are called the \emph{proposed configuration}. They improve some measures but did not pass every path-preservation check. The results therefore describe a possible numerical upgrade, not a universally better or physically validated replacement.

%============================================================
% 5.2 NUMERICAL EVALUATION METHODOLOGY
%============================================================
\section{Numerical Evaluation Methodology}
\label{sec:numerical_methodology}

%============================================================
% 5.2.1 MEASURED HARDWARE AND TUBE-LENGTH MAPPING
%============================================================
\subsection{Measured Hardware and Tube-Length Mapping}
The rear actuator holds the inner element, the middle actuator holds the middle tube, and the front actuator holds the outer tube. Distances were measured from the front of each brass chuck to the outside front face of the front plate. The measured distance bounds are listed in Table~\ref{tab:measured_carriage_bounds}.

@@tab:measured_carriage_bounds@@

Each actuator has a nominal 100~mm stroke and a full axial length of 71.5~mm, including the rear shaft protrusion. The permitted gap between a chuck tip and the rear of the actuator ahead is approximately 8.5--108~mm. A maximum gap of 108.5~mm was also checked to assess the 0.5~mm discrepancy between the measured gap range and the nominal relative travel.

For chuck-to-tip material length $L_i$ and chuck-to-plate distance $d_i$, the exposed material length is
\begin{equation}
e_i=L_i-d_i.
\label{eq:exposure_mapping}
\end{equation}
No grip offset is added. The specified tube lengths are provisionally treated as chuck-front-to-tip lengths because the amount retained behind the grip has not been confirmed. Exposed material length follows the tube backbone; it is not the Cartesian distance from the plate to the tip.

The exterior model requires $e_\mathrm{inner}\geq e_\mathrm{middle}\geq e_\mathrm{outer}\geq0$. This excludes tips retracted behind the front plate or behind the tip of the surrounding tube. These are model restrictions as well as operating assumptions. Negative exposures are rejected rather than set to zero.

In inner/middle/outer order, the adjacent actuator gaps must satisfy
@@eq:coupled_clearance@@
for each adjacent pair, alongside the individual distance bounds in Table~\ref{tab:measured_carriage_bounds}. All constraints are applied together.

The original inner, middle and outer exposure ranges are 75--175, 0--75 and 0--65~mm, respectively. The proposed ranges are 75--175, 0--82.5 and 0--72.5~mm. These give each element's available range across feasible configurations; they do not mean that every combination is allowed. Both designs retain a maximum middle/front gap of 18.5~mm within the exterior model because the difference between their middle and outer total lengths remains 90~mm.

Straight internal routing, straightening at the plate exit and unrestricted rotation are assumed. Internal guidance, friction, distributed torsion, elastic stability, anatomical collision and behaviour under load are not evaluated.

%============================================================
% 5.2.2 SAMPLING AND WORKSPACE ESTIMATION
%============================================================
\subsection{Sampling and Workspace Estimation}
\label{subsec:workspace_method}
Each configuration was assessed using 262,144 feasible actuator states. Scrambled Sobol sequences sampled the measured carriage ranges and uniformly distributed rotations, with states outside the coupled limits rejected. Both designs used the same sequence seed. However, their different feasible domains meant that the accepted actuator states were not identical.

Tip positions were grouped by radial distance $r$ and axial position $z$ using the ideal model's axial symmetry. The primary grid used 5~mm radial--height cells. Each occupied cell was assigned an annular volume:
@@eq:swept_cell_volume@@
Here, $r_{c,\mathrm{in}}$ and $r_{c,\mathrm{out}}$ are the cell's radial boundaries and $\Delta z$ is its height. Summing these volumes gives an occupied-workspace estimate. It does not prove that every point within an occupied cell or a displayed envelope is reachable. Rotated copies used for visualisation were not counted as additional samples.

The original configuration's primary occupied cells were fixed as the reference for the dense IK comparison. Table~\ref{tab:evaluation_settings} separates the historical study settings from the current comparison.

@@tab:evaluation_settings@@

%============================================================
% 5.2.3 POSITIONAL DEXTERITY
%============================================================
\subsection{Positional Dexterity}
\label{subsec:dexterity_method}
Positional dexterity was assessed using the positional Jacobian $\mathbf J$, which relates small actuator changes to changes in tip position. Translation and rotation were scaled using the same reference values for both tube designs:
@@eq:normalised_jacobian@@
Translation derivatives use millimetres and rotation derivatives use radians. The straight inner element's rotation does not change its intrinsic shape in this model. Positional isotropy was calculated as
@@eq:positional_isotropy@@
where $\sigma_{\min}$ and $\sigma_{\max}$ are the smallest and largest singular values of the scaled Jacobian. Higher isotropy indicates a more balanced local positional response under this scaling. It does not measure stiffness, orientation control or movement available in every direction at an actuator limit.

The maps show the median isotropy in each cell. Comparisons of regional means use common cells with at least 30 sampled states per design, weighted by their annular volumes. Spatial P10 is the lower volume-weighted decile of these cell medians. These summaries differ from the earlier search scores, which used a five-sample support rule and assigned zero to unsupported reference cells.

%============================================================
% 5.2.4 FIXED-START INVERSE-KINEMATICS EVALUATION
%============================================================
\subsection{Fixed-Start Inverse-Kinematics Evaluation}
\label{subsec:fixed_start_method}
Fresh target positions were generated using the original configuration's forward model, independently of the candidate-selection samples. Targets retained their full azimuth, and both designs received identical Cartesian coordinates. The actuator states used to generate them were saved but were not supplied to the solver.

Up to 192 targets were retained per reference cell. The initial target bank did not meet the local support requirement, so six additional batches of independent original-model samples were used to increase coverage. This addition depended only on target counts, not on the proposed configuration's errors. The regional definitions remained fixed and the initial results were archived.

The numerical residual was the distance between the target $\mathbf p^*$ and the solved tip position:
@@eq:numerical_ik_error@@
A residual of at most 0.5~mm counted as success. Median, P95, mean and maximum residuals included failed attempts. P95 describes the 95th percentile; it is not a maximum or a physical accuracy guarantee.

Both designs used the existing damped least-squares solver with 40 iterations, 1~mm damping, a normalised component step limit of 0.1, a finite-difference step of $10^{-4}$ and a stopping residual of 0.01~mm. Every target started from exposed lengths of 75/0/0~mm in inner/middle/outer order and zero rotations. No restarts or target-derived initial guesses were used. The deployment parameterisation enforced the measured coupled constraints.

Each target was weighted by its cell volume divided by the number of targets in that cell. Regional percentiles were calculated from the weighted target residuals. Although the model geometry is axially symmetric, a solver starting at zero rotations can perform differently across target directions. The current full-azimuth success rates are therefore not directly comparable with earlier canonical-plane evaluations.

%============================================================
% 5.2.5 VERIFICATION, SUPPORT AND REGIONAL DEFINITIONS
%============================================================
\subsection{Verification, Support and Regional Definitions}
\label{subsec:numerical_support}
Median maps require at least 30 states or targets per cell, depending on the measure. P95 and failure-rate maps require at least 100 targets. The final shared target bank met this requirement over 90.48\% of the original reference volume, exceeding the predefined 90\% goal. Cells with insufficient support are masked rather than treated as successful. Global target summaries retain all available attempts.

The vectorised evaluation solver was checked against the native viewer solver on 12 targets per design. Their residuals differed by less than $4.5\times10^{-11}$~mm. This checks implementation agreement, not physical model accuracy. Workspace estimates were also compared across sample counts, 2.5/5/10~mm grids and an independent 131,072-state repeat.

For the proposed configuration, changing the maximum gap from 108 to 108.5~mm produced identical sampled states. Tube nesting already restricted the gap before this upper stop became active. This result applies to the checked design and should not be assumed for other tube geometries.

Four regions were used to examine where performance changed:
\begin{enumerate}
\item The forward corridor: cells with $r<10$~mm and $40\leq z<130$~mm.
\item The low-dexterity region: the lowest 20\% of the reference volume by dexterity from the earlier measured original-tube study, mapped to the current grid.
\item The IK-problem region: cells with at least 100 independent original-only calibration targets and at least 10\% fixed-start failures. This used a separate 38,558-target calibration set.
\item The remaining original reference cells outside the union of these regions.
\end{enumerate}
The first three regions can overlap, so their volumes and target counts cannot be added. The remaining region includes some poorly supported boundary cells and is not assumed to be easier to control.

Figures~\ref{fig:ik_target_support} and~\ref{fig:numerical_sampling_support} show the target support and sensitivity checks.

@@fig:ik_target_support@@

@@fig:numerical_sampling_support@@

\FloatBarrier

%============================================================
% 5.3 CHARACTERISATION OF THE ORIGINAL CONFIGURATION
%============================================================
\section{Characterisation of the Original Configuration}
\label{sec:original_characterisation}
The earlier original-model study estimated an occupied workspace of 48,656.99~cm$^3$ on a 10~mm grid. Increasing the sample from 150,000 to 200,000 states changed the estimate by approximately 0.15\%, while an independent sample covered 98.51\% of the reference occupancy. Figure~\ref{fig:original_model_workspace} retains this result under its original tube-length-based deployment bounds.

@@fig:original_model_workspace@@

The historical 100,000-target evaluation reported 91.96\% volume-weighted success, a median residual of 0.000378~mm and a P95 residual of 217.961~mm. Local P95 and failure-rate support covered 91.15\% of its reference volume. Most successful solves had small residuals, but a minority of failed solves produced large errors.

These results describe the earlier model domain, not the measured platform's demonstrated capability. The present baseline uses different movement limits, initialization, target directions and grid size. The historical values are retained as context and are not used to calculate the proposed configuration's improvement.

%============================================================
% 5.4 OPTIMISATION AND CONFIGURATION SELECTION
%============================================================
\section{Optimisation and Configuration Selection}
\label{sec:optimisation_method}
The search varied total tube lengths, middle and outer curved lengths, and their pre-curvatures. Diameters and nominal Young's modulus remained fixed, and the inner element remained straight. The original physical tube specification was the reference throughout.

An earlier search did not identify a configuration that improved every tested metric simultaneously. The subsequent search therefore prioritised fixed-start reaching in the low-dexterity region and forward corridor, supported by lower-tail isotropy. Workspace volume and maximum reach were secondary, with predefined allowable losses of 5\%. Further limits protected reference-cell retention, global isotropy and reaching performance. Weak-region and corridor success gains received twice the ranking weight of global success gains. These were numerical design preferences rather than clinical requirements.

The search screened 115 valid designs, including references. Of these, 26 received denser spatial assessment, including references, and eight candidates received path assessment. Two candidates passed the training checks. For the inner, middle and outer elements, the first-ranked design used total lengths of 350, 177.5 and 87.5~mm. Their pre-curvatures were 0, 21.37 and 14.04~m$^{-1}$, respectively. Its parameters were fixed before evaluation on two independent target and path banks.

The first candidate failed the held-out path-preservation rule. The second candidate was then evaluated on separate fresh banks and also failed that rule. The first candidate is retained here as a proposed point-to-point upgrade with explicit trade-offs. It is not presented as having passed all acceptance criteria.

The following dense maps assess this fixed first candidate using new samples and no further parameter changes. They supplement the earlier evaluation without replacing its path results.

%============================================================
% 5.5 ORIGINAL VERSUS PROPOSED CONFIGURATION
%============================================================
\section{Original versus Proposed Configuration}
\label{sec:configuration_comparison}

%============================================================
% 5.5.1 PARAMETER AND SHAPE CHANGES
%============================================================
\subsection{Parameter and Shape Changes}
\label{subsec:parameter_comparison}
Table~\ref{tab:optimised_parameters} lists both tube configurations. The middle and outer total lengths each increase by 7.5~mm, while middle pre-curvature increases by approximately 11.77\%. The inner element, tube diameters, material modulus and curved lengths remain unchanged.

@@tab:optimised_parameters@@

Figure~\ref{fig:optimised_tube_shapes} shows the corresponding intrinsic shapes before the tubes are combined.

@@fig:optimised_tube_shapes@@

The proposed middle tube has a 90~mm curved section but a maximum exposure of 82.5~mm. At the same total length and curvature, the exterior model cannot distinguish between middle curved lengths at or above this exposure limit. The selected curved length is therefore not uniquely optimal. Retention, guide passage and where bending can physically begin still require confirmation.

The proposed configuration was also placed in a separate interactive viewer, shown in Figure~\ref{fig:workspace_simulator}. The display includes relative actuator positions and a sampled workspace envelope. Its 12,000-state envelope is for visualisation; the denser samples described above provide the quantitative comparison.

@@fig:workspace_simulator@@

\FloatBarrier

%============================================================
% 5.5.2 WORKSPACE COMPARISON
%============================================================
\subsection{Workspace Comparison}
\label{subsec:workspace_comparison}
The primary occupied-workspace estimate increased from 2396.25 to 2582.78~cm$^3$, a gain of 7.78\%. The independent repeat gave 2313.39 and 2521.52~cm$^3$, respectively. Both samples therefore showed a larger proposed workspace, although the estimated volume depends on sample density and cell size.

Figures~\ref{fig:workspace_comparison} and~\ref{fig:workspace_overlay} show where occupancy changed. The proposed design gained 216.77~cm$^3$ of sampled occupancy, while 30.24~cm$^3$ of the original occupancy was absent. It did not preserve every original occupied cell.

@@fig:workspace_comparison@@

@@fig:workspace_overlay@@

\FloatBarrier

%============================================================
% 5.5.3 POSITIONAL DEXTERITY COMPARISON
%============================================================
\subsection{Positional Dexterity Comparison}
\label{subsec:isotropy_comparison}
The volume-weighted mean of cell-median isotropy increased from 0.34997 to 0.37396, or 6.86\%, across common supported original-reference cells. These cells covered 79.92\% of the original reference volume. Spatial P10 also increased, from 0.19716 to 0.21290.

The forward corridor showed a larger mean increase, from 0.17757 to 0.20871, or 17.53\%. Figures~\ref{fig:isotropy_comparison} and~\ref{fig:isotropy_difference} show the distribution of these changes. The difference map separates local gains and losses rather than relying only on the overall mean.

@@fig:isotropy_comparison@@

@@fig:isotropy_difference@@

\FloatBarrier

%============================================================
% 5.5.4 FIXED-START IK COMPARISON
%============================================================
\subsection{Fixed-Start IK Comparison}
\label{subsec:ik_comparison}
Both designs were evaluated on the same 59,341 fresh targets. Volume-weighted success increased from 62.280\% to 66.993\%, a gain of 4.713 percentage points. Median, mean and P95 residuals decreased, but the maximum residual increased from 116.426 to 118.113~mm. Table~\ref{tab:fixed_start_comparison} gives the full comparison, including failed attempts.

@@tab:fixed_start_comparison@@

Figures~\ref{fig:ik_median_comparison}--\ref{fig:ik_failure_comparison} show the median residual, P95 residual and failure rate across the original reference workspace. Blue indicates lower residuals or failure rates, while red indicates higher values. Both designs use the same scale for each measure. The median and P95 maps also share the same residual scale, so their colours can be compared directly. Grey cells have insufficient local support and should not be interpreted as low-error regions.

@@fig:ik_median_comparison@@

@@fig:ik_p95_comparison@@

@@fig:ik_failure_comparison@@

\FloatBarrier

%============================================================
% 5.5.5 REGIONAL COMPARISON
%============================================================
\clearpage
\subsection{Regional Comparison}
\label{subsec:regional_comparison}
Tables~\ref{tab:regional_isotropy} and~\ref{tab:regional_ik} compare the frozen reference regions. Isotropy means use common cells with sufficient state samples. IK statistics use the available weighted target attempts in each region. These summaries therefore use different support criteria.

@@tab:regional_isotropy@@

@@tab:regional_ik@@

Forward-corridor success increased from 67.49\% to 74.79\%, a gain of 7.30 percentage points. Success increased by 4.00 percentage points in the low-dexterity region and 6.01 percentage points in the IK-problem region. The remaining region decreased from 53.92\% to 53.15\%, a loss of 0.78 percentage points.

The supported isotropy means increased in all four regions. However, only 14.83\% of the remaining region had common 30-state support. Its isotropy mean describes that subset and cannot represent the entire region. The results show improvements in the prioritised regions alongside a small reaching loss elsewhere. Higher local isotropy also did not guarantee successful fixed-start IK.

\FloatBarrier

%============================================================
% 5.6 PATH-FOLLOWING EVALUATION
%============================================================
\section{Path-Following Evaluation}
\label{sec:path_diagnosis}
The independent path evaluation used two banks of 24 original-reference trajectories. The original configuration completed 44 of 48 paths, compared with 43 for the proposed configuration. A path counted as complete only if all 81 checked deviations were at most 0.5~mm. The average of the two banks' P95 tracking residuals increased from 0.087 to 0.153~mm. This is an average of two percentiles, not a pooled P95 or a maximum error.

The one-path reduction was a net change. The proposed tubes completed two paths that the original tubes failed, but failed three that the original tubes completed. Two new failures reached maximum deviations of 12.05 and 37.84~mm.

These two paths were investigated further. Alternative initial tube configurations at essentially the same starting tip positions reduced the densely checked maximum deviations to 0.027 and 0.115~mm. The tube geometry, measured hardware limits, 20 motion steps and 40-iteration controller budget were unchanged. This indicates that the starting configuration affected performance in these cases.

The detailed investigation is presented in Appendix~\ref{app:path_recovery}. These recovery tests do not replace the original 43/48 completion result, as the alternative starts were selected after examining the failures. An automatic method for selecting and reaching those starts was not evaluated.

%============================================================
% 5.7 PRINCIPAL FINDINGS
%============================================================
\section{Principal Findings}
\label{sec:numerical_findings}
The measured carriage and clearance limits reduced the available model domain and required the actuators to be treated as coupled. Under the same hardware constraints, the proposed tubes increased the primary sampled workspace estimate by 7.78\% and global fixed-start success by 4.713 percentage points. The forward corridor showed larger improvements in positional isotropy and reaching success.

These gains were not uniform. Some original workspace occupancy was lost, and reaching success in the remaining region decreased slightly. Path completion fell from 44/48 to 43/48. The recovery checks identified lower-error alternatives for two failed paths, but did not demonstrate a general initialization method.

The proposed configuration is therefore a possible numerical point-to-point upgrade with documented trade-offs. Sampling dependence, provisional installed lengths, unresolved guidance and the lack of physical validation limit the conclusions. Their practical significance is discussed in Chapter~\ref{chap:discussion}.
\FloatBarrier
'''

chapter = expand(chapter)
# Chapter 6 may not have this label in the user's full document. Use a plain
# chapter number here to keep this standalone package independent of Chapter 6.
chapter = chapter.replace(r'Chapter~\ref{chap:discussion}', 'Chapter~6')
chapter = chapter.replace(
    'The shared scale is linear below 0.01 mm and logarithmic above it; the same normalization is used in the P95 map.',
    'Blue indicates lower residuals and red higher residuals. The shared scale is linear below 0.01 mm and logarithmic above it; the P95 map uses the same scale.')
chapter = chapter.replace(
    'The same colour scale as the median map is used. Failed attempts are retained;',
    'Blue indicates lower residuals and red higher residuals, using the same scale as the median map. Failed attempts are retained;')
chapter = chapter.replace(
    'Failure indicates the outcome of this initialization and solver, not proof of geometric unreachability.',
    'Blue indicates lower failure rates and red higher rates. Failure describes this initialization and solver, not proof of geometric unreachability.')
(TEX / 'chapter5.tex').write_text(chapter)

appendix = r'''%============================================================
% APPENDIX: PATH-FOLLOWING FAILURE INVESTIGATION
%============================================================
% Include after \appendix in the main document, following any other appendices.
% Do not put \appendix here: the main document controls appendix numbering.
\chapter{Path-Following Failure Investigation}
\label{app:path_recovery}

%============================================================
% PURPOSE OF THE INVESTIGATION
%============================================================
\section{Purpose of the Investigation}
Two proposed-tube trajectories developed maximum deviations of 12.05 and 37.84~mm under the original initialization procedure. The investigation checked whether different starting tube configurations could follow these paths under the same hardware limits.

In the first case, the outer tube reached its maximum exposure of 72.5~mm. In the second, it reached zero exposure, which is a boundary of the exterior model rather than the physical rear carriage stop. The solver then made little progress as the reference path continued. Increasing the iteration budget or reducing the motion step size alone did not resolve these starts.

%============================================================
% ALTERNATIVE STARTING CONFIGURATIONS
%============================================================
\section{Alternative Starting Configurations}
Independent pointwise solves and reverse continuation identified alternative tube arrangements at essentially the same starting tip positions. Reverse continuation follows the reference backwards from a separately solved endpoint to find another starting arrangement. The path was then followed forwards from that arrangement.

With the same 20 motion steps and 40-iteration controller budget, the maximum deviations decreased to 0.027 and 0.115~mm when checked at 2,001 positions per trajectory. All checked actuator states respected the measured coupled constraints. The lower-error results demonstrate feasible alternatives at the checked positions within the ideal model; they do not prove continuous-path feasibility between every pair of checks.

Figure~\ref{fig:path_recovery} compares the paths and their deviations. The deviation plots use a logarithmic vertical scale, where equal intervals represent multiplicative changes. The dashed line marks the 0.5~mm threshold. ``Original start'' means the initially tested posture of the proposed tubes, not the original tube design.

%============================================================
% INTERPRETATION AND LIMITATIONS
%============================================================
\section{Interpretation and Limitations}
The alternative starts were identified after the failures had been examined. They therefore remain recovery checks and do not replace the original 43/48 path-completion result. The investigation did not implement an automatic initialization method or plan the transition from a prescribed current posture to the alternative start.

The results indicate that starting configuration and behaviour near deployment limits affected these two failures. They do not establish that every failed trajectory can be recovered. A subsequent controller comparison would need to apply the same automatic strategy to both tube designs and assess fresh paths.

@@fig:path_recovery@@
\FloatBarrier
'''
(TEX / 'appendix_path_recovery.tex').write_text(expand(appendix))
main = (SOURCE / 'overleaf/main.tex').read_text().replace(
    r'\end{document}', '\\appendix\n\\input{appendix_path_recovery}\n\\end{document}')
(TEX / 'main.tex').write_text(main)

# Re-render only the three requested figures, from the archived map arrays.
# All spatial bins, support thresholds and numerical scales are unchanged.
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none'})
names = ['original', 'proposed']
maps = {n:dict(np.load(SOURCE/'data'/f'{n}_ik_maps.npz')) for n in names}
ref = np.array(json.loads((SOURCE/'data/reference.json').read_text()))
shape = (38,25)
extent = [0,125,0,190]
def array(ids, values):
    result=np.full(shape,np.nan)
    ids=np.asarray(ids,int)
    result[ids%1000,ids//1000]=values
    return result

maxerr=max(float(maps[n]['p95'][maps[n]['counts']>=100].max()) for n in names)
norm=SymLogNorm(linthresh=.01,linscale=1,vmin=0,vmax=maxerr)
for metric, support, title in [
    ('median',30,'Median fixed-start IK residual'),
    ('p95',100,'95th-percentile fixed-start IK residual'),
    ('failure',100,'Fixed-start failure fraction | residual > 0.5 mm')]:
    fig,axs=plt.subplots(1,2,figsize=(8.4,4.9),sharex=True,sharey=True,layout='constrained')
    for ax,n in zip(axs,names):
        m=maps[n]
        ax.imshow(array(ref,np.ones(len(ref))),origin='lower',extent=extent,aspect='auto',
                  cmap=ListedColormap(['#b6bcc2']),vmin=0,vmax=1,interpolation='nearest')
        keep=m['counts']>=support
        values=m[metric]*(100 if metric=='failure' else 1)
        scale={'vmin':0,'vmax':100} if metric=='failure' else {'norm':norm}
        im=ax.imshow(array(m['cells'][keep],values[keep]),origin='lower',extent=extent,
                     aspect='auto',cmap='RdYlBu_r',interpolation='nearest',**scale)
        ax.set(xlim=(0,125),ylim=(0,190),xlabel='Radial distance (mm)',
               ylabel='Z from outside front plate (mm)',title=n.title())
        ax.set_facecolor('white')
    fig.suptitle(title+'\nBlue: lower (better) | Red: higher (worse)',fontsize=11)
    fig.colorbar(im,ax=axs,label='Targets exceeding threshold (%)' if metric=='failure'
                 else 'Numerical residual (mm)',shrink=.85)
    for ext in ['png','svg']:
        fig.savefig(FIG/f'ik_{metric}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

manifest={
    'source_chapter':str(SOURCE/'overleaf/chapter5.tex'),
    'changes':['Plain-language rewrite; numbered comment headers',
               'Recovery figure and detailed diagnosis moved to separate appendix',
               'IK maps: blue low / red high with explicit direction legends'],
    'new_simulation_runs':False,
    'data_sources':{},
    'maps':{'colormap':'RdYlBu_r','residual_min_mm':0,'residual_max_mm':maxerr,
            'residual_normalisation':'SymLogNorm(linthresh=0.01,linscale=1)',
            'failure_range_pct':[0,100],'support_median':30,'support_p95_failure':100},
}
for file in [SOURCE/'overleaf/chapter5.tex',SOURCE/'data/reference.json',
             SOURCE/'data/original_ik_maps.npz',SOURCE/'data/proposed_ik_maps.npz']:
    manifest['data_sources'][str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
(OUT/'revision_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Revised sources and figures:',TEX)

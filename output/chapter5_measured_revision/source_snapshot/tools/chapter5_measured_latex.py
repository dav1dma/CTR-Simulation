"""Build a self-contained Overleaf chapter from source evidence and dense results."""
from pathlib import Path
import json,shutil,re,zipfile,hashlib
import numpy as np
import chapter5_measured_study as s
OUT=s.OUT;DATA=s.DATA;PKG=OUT/'overleaf';PKG.mkdir(exist_ok=True)
summary=json.loads((DATA/'summary.json').read_text());sp=json.loads((DATA/'spatial_regions.json').read_text());ov=json.loads((DATA/'overlay.json').read_text());checks=json.loads((DATA/'sampling_checks.json').read_text());v=json.loads((DATA/'solver_verification.json').read_text());top=json.loads((DATA/'support_topup_log.json').read_text()) if (DATA/'support_topup_log.json').exists() else []
a=summary['original'];b=summary['proposed'];ga=a['global_ik'];gb=b['global_ik'];support=a['p95_supported_reference_fraction']*100
source=Path('/Users/david/.codex/attachments/396b7632-a960-4bcf-a10c-ad214ccb460e/pasted-text.txt');shutil.copy2(source,OUT/'chapter5_supplied_original.tex')
def figure(file,caption,label,width='\\linewidth'):
 return '\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width='+width+']{ch5/'+file+'}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{figure}\n'
def table(caption,label,columns,head,rows):
 return '\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'+head+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
figs={
'WORKSPACE':figure('workspace.png','Original and proposed occupied radial--height cells under the same measured hardware constraints, using 262,144 feasible states per design and 5 mm cells. Sampling sequences have the same seed, but the design-dependent feasible domains mean the accepted configurations are not identical.','fig:workspace_comparison'),
'OVERLAY':figure('workspace_overlay.png',f"Shared occupied cells (grey), proposed-only cells (green), and original-only cells (orange). Gained and original-only annular volumes are {ov['gained']:.2f} and {ov['lost']:.2f} cm$^3$. These are sample-dependent occupancy differences, not exact geometric set differences. The dashed box marks the forward corridor.",'fig:workspace_overlay','0.8\\linewidth'),
'ISO':figure('isotropy_median.png','Median positional isotropy with a common colour scale. Each displayed cell contains at least 30 sampled configurations. Grey denotes occupied cells with insufficient support; white denotes cells outside the sampled occupancy.','fig:isotropy_comparison'),
'DIFF':figure('dexterity_difference.png','Proposed minus original cell-median isotropy in common cells with at least 30 states per design. Blue indicates higher isotropy and red lower isotropy. Grey denotes insufficient common support. The enlarged corridor retains the same colour scale. Differences are absolute isotropy units, not percentages.','fig:isotropy_difference'),
'MEDIAN':figure('ik_median.png','Median fixed-start numerical residual on identical targets. Each displayed cell has at least 30 targets. The shared scale is linear below 0.01 mm and logarithmic above it; the same normalization is used in the P95 map. Grey indicates insufficient support inside the original reference occupancy.','fig:ik_median_comparison'),
'P95':figure('ik_p95.png','Cell-wise 95th-percentile fixed-start residual. Each displayed cell contains at least 100 targets. The same colour scale as the median map is used. Failed attempts are retained; local P95 is neither a maximum nor a confidence bound.','fig:ik_p95_comparison'),
'FAILURE':figure('ik_failure.png','Fraction of fixed-start attempts exceeding 0.5 mm residual, using identical targets and at least 100 targets per displayed cell. Failure indicates the outcome of this initialization and solver, not proof of geometric unreachability.','fig:ik_failure_comparison'),
'SUPPORT':figure('target_support.png',f"Number of shared targets per cell. Cells with at least 100 targets cover {support:.2f}\\% of the frozen original reference volume. Sparse boundary cells remain masked in local P95 and failure maps.",'fig:ik_target_support','0.8\\linewidth'),
'CONVERGENCE':figure('sampling_support.png','Sensitivity of occupied-volume estimates to sample count and cell size, and reference-volume retention by an independent 131,072-state repeat. Differences reflect discretization and sampling; they do not establish exact geometric convergence.','fig:numerical_sampling_support'),
'SHAPES':figure('tube_shapes.png','Intrinsic original and proposed tube shapes shown separately, with a straight section followed by a distal constant-curvature arc. The inner element is unchanged. The proposed middle and outer tubes are longer; only middle precurvature changes. Curved lengths remain unchanged. These are ideal geometric drawings, not physical measurements.','fig:optimised_tube_shapes'),
'VIEWER':figure('proposed_viewer.png','Separate interactive viewer for the proposed configuration, including a schematic actuator-position display and an approximate envelope from 12,000 feasible states. This display is distinct from the denser maps used for quantitative evaluation. The existing controller is retained.','fig:workspace_simulator'),
'HISTORY':figure('historical_workspace.png','Archived original-model workspace under tube-length-based deployment bounds. It characterises the earlier model domain, not the measured actuator workspace used in the present comparison. Display rotations do not increase the independent sample count.','fig:original_model_workspace','0.8\\linewidth'),
'PATH':figure('path_recovery.png','Post-hoc diagnosis of two failed proposed-tube trajectories. Alternative configurations at essentially the same initial tip positions reduce densely checked maximum deviations from 12.05 and 37.84 mm to 0.027 and 0.115 mm. Geometry, hardware limits, 20 motion steps and 40-iteration budget are unchanged. These selected recovery cases do not replace the original path-completion score.','fig:path_recovery')}
rows=[['Success within 0.5 mm (\\%)',f"{ga['success_pct']:.3f}",f"{gb['success_pct']:.3f}"],['Median residual (mm)',f"{ga['median_mm']:.6f}",f"{gb['median_mm']:.6f}"],['P95 residual (mm)',f"{ga['p95_mm']:.3f}",f"{gb['p95_mm']:.3f}"],['Mean residual (mm)',f"{ga['mean_mm']:.3f}",f"{gb['mean_mm']:.3f}"],['Maximum residual (mm)',f"{ga['max_mm']:.3f}",f"{gb['max_mm']:.3f}"]]
figs['IKTABLE']=table('Volume-weighted fixed-start comparison on identical fresh targets. Failed attempts are included. The maximum is the largest observed target residual.','tab:fixed_start_comparison','lrr','Metric & Original & Proposed',rows)
region_names={'corridor':'Forward corridor','weak':'Low-dexterity region','ik_problem':'IK-problem region','remaining':'Remaining region'}
rows=[]
for key,label in region_names.items():
 x,y=sp[key]['original'],sp[key]['proposed'];rows.append([label,f"{x['mean']:.5f}",f"{y['mean']:.5f}",f"{100*(y['mean']/x['mean']-1):+.2f}",f"{sp[key]['support_fraction']*100:.1f}"])
figs['ISOTABLE']=table('Regional means of cell-median isotropy on common cells with at least 30 states per design. Support is the fraction of each original reference region represented. Regions overlap.','tab:regional_isotropy','lrrrr','Region & Original & Proposed & Change (\\%) & Support (\\%)',rows)
rows=[]
for key,label in region_names.items():
 x,y=a['regions'][key],b['regions'][key];rows.append([label,f"{x['success_pct']:.2f}",f"{y['success_pct']:.2f}",f"{x['p95_mm']:.3f}",f"{y['p95_mm']:.3f}"])
figs['REGIK']=table('Regional fixed-start success and weighted P95 on the same target bank. O: original; P: proposed. Regional percentiles are computed from target residuals, not by averaging cell percentiles.','tab:regional_ik','lrrrr','Region & O success (\\%) & P success (\\%) & O P95 (mm) & P P95 (mm)',rows)
text=r'''% Revised Chapter 5: measured coupled hardware; original source preserved separately.
\chapter{Numerical Studies}
\label{chap:numerical_studies}

\section{Study Objectives and Scope}
\label{sec:numerical_scope}
The numerical studies investigated the original tube configuration and a proposed tube-geometry revision within the same measured CTR platform. The comparison addressed how actuator limits shape the workspace, where positional dexterity is weak, and whether the proposed tubes improve target reaching while retaining useful capability elsewhere.

The original tubes remained the physical reference. An earlier study characterised the original model using tube-length-based deployment bounds. The present comparison instead uses the measured carriage travel and coupled clearances for both designs. Its results supersede the earlier estimated-hardware comparison that assumed a 35 mm gripping offset. Changing the hardware model is distinct from changing the tube geometry, and gains are calculated only between designs evaluated under the same current assumptions.

The revised tubes are termed the \emph{proposed configuration}. They were selected through a bounded numerical search but did not pass every path-preservation criterion. They are therefore not presented as a globally optimal or universally superior replacement. All results are ideal-model predictions; physical accuracy, anatomical compatibility and clinical performance remain unverified.

\section{Numerical Evaluation Methodology}
\label{sec:numerical_methodology}
\subsection{Measured Hardware and Tube-Length Mapping}
The rear, middle and front actuators hold the inner, middle and outer elements respectively. Distances are measured from the brass chuck front to the outside front face of the front plate. Each actuator has a nominal 100 mm stroke. Its full axial length, including the rear shaft protrusion, is 71.5 mm. Adjacent chuck-to-actuator-rear gaps must remain between 8.5 and 108 mm; 108.5 mm is retained as a sensitivity case.

\begin{table}[htbp]
\centering\small
\caption{Measured chuck-to-front-plate distance bounds.}
\label{tab:measured_carriage_bounds}
\begin{tabular}{lrr}\toprule
Actuator/element & Retracted (mm) & Extended (mm)\\\midrule
Rear/inner & 275 & 175\\
Middle/middle & 195 & 95\\
Front/outer & 115 & 15\\\bottomrule
\end{tabular}
\end{table}

For chuck-to-tip material length $L_i$ and chuck-to-plate distance $d_i$, exterior material length is $e_i=L_i-d_i$. No 35 mm grip offset is applied. Material lengths retain a provisional chuck-front-to-tip interpretation because retained material behind the grip has not been established. The exterior model requires $e_\mathrm{inner}\geq e_\mathrm{middle}\geq e_\mathrm{outer}\geq0$. Signed negative exposures are excluded rather than converted to valid states by clipping.

In inner/middle/outer order, individual distance bounds and the adjacent constraints
\begin{equation}
8.5\leq d_i-d_{i+1}-71.5\leq108\quad\mathrm{mm}
\label{eq:coupled_clearance}
\end{equation}
must hold simultaneously. Feasible exposure bounds are 75--175/0--75/0--65 mm for the original tubes and 75--175/0--82.5/0--72.5 mm for the proposed tubes. These are marginal bounds, not independently available strokes. Both designs retain an exterior-model middle/front gap limit of 18.5 mm because their middle-minus-outer total-length difference remains 90 mm.

Straight internal routing, effective straightening at the plate exit and unrestricted rotations are assumed. Internal guidance, friction, distributed torsion, elastic stability, anatomical collision and loaded accuracy are not evaluated.

\subsection{Sampling and Workspace Estimation}
\label{subsec:workspace_method}
Each design was evaluated at 262,144 feasible configurations generated by rejection sampling from the measured carriage box and uniformly distributed rotations using scrambled Sobol sequences. Both designs used the same sequence seed; their different admissible domains mean their accepted configurations were not identical. Workspace positions were represented by radial distance $r$ and axial coordinate $z$, exploiting the ideal model's axial symmetry. Rotated display copies were not counted as additional samples.

The primary grid used 5 mm radial--height cells. Occupied cells were assigned swept annular volumes
\begin{equation}
V_c=\pi(r_{c,\mathrm{out}}^2-r_{c,\mathrm{in}}^2)\Delta z.
\label{eq:swept_cell_volume}
\end{equation}
Their sum estimates occupied workspace volume, but does not prove that every enclosed point is reachable. The original primary occupancy was frozen as the spatial reference before the dense IK comparison.

\begin{table}[htbp]
\centering\small
\caption{Historical and current evaluation settings. Their absolute scores are not directly comparable.}
\label{tab:evaluation_settings}
\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}\toprule
Setting & Historical characterisation & Measured-hardware comparison\\\midrule
Designs & Original & Original and proposed\\
Movement domain & Tube-length-based bounds & Measured coupled carriage limits\\
Primary grid & 10 mm & 5 mm\\
Workspace sample & 200,000 training; 50,000 validation & 262,144 per design; independent 131,072-state repeat\\
Main IK targets & 100,000 & @@NTARGET@@ identical targets per design\\
IK initialization & Zero exposure and rotations & 75/0/0 mm exposure; zero rotations\\
Target convention & Historical canonical plane & Actual full-azimuth original-model positions\\\bottomrule
\end{tabularx}
\end{table}

\subsection{Positional Dexterity}
\label{subsec:dexterity_method}
Positional dexterity was assessed using a range-normalised positional Jacobian, with the scaling fixed for both designs:
\begin{equation}
\mathbf S=\operatorname{diag}(350,170,80,\pi,\pi,\pi),\qquad \widetilde{\mathbf J}=\mathbf J\mathbf S.
\label{eq:normalised_jacobian}
\end{equation}
Physical deployment derivatives use millimetres and rotation derivatives radians. The straight inner element has no active intrinsic rotation effect in this model. Isotropy is
\begin{equation}
\eta=\frac{\sigma_{\min}(\widetilde{\mathbf J})}{\sigma_{\max}(\widetilde{\mathbf J})}.
\label{eq:positional_isotropy}
\end{equation}
Higher values indicate more balanced local positional response under this normalization. They do not establish stiffness, orientation dexterity, speed capability or feasible motion in every direction at a stop.

Maps show median isotropy within each cell. The updated regional summaries use the same common cells with at least 30 configurations per design and weight cell medians by swept volume. Spatial P10 denotes the lower volume-weighted decile of these medians. These descriptive results are distinct from the search-stage scores, which used a five-sample rule and assigned zero to unsupported reference cells.

\subsection{Fixed-Start Inverse-Kinematics Evaluation}
\label{subsec:fixed_start_method}
Fresh targets were generated by original-design forward kinematics, independently of configuration selection. Their full azimuth was retained. The original and proposed designs received identical target coordinates; generating actuator states were saved for reproducibility but not supplied to the solver. Up to 192 targets were retained per reference cell. @@TOPUP@@

The numerical residual was
\begin{equation}
\varepsilon=\lVert\mathbf p^*-\mathbf p(\mathbf q_\mathrm{solved})\rVert_2.
\label{eq:numerical_ik_error}
\end{equation}
A residual at most 0.5 mm counted as success. Median, P95, mean and maximum residuals were recorded with failed attempts retained. P95 is not a maximum, confidence bound or safety guarantee.

Both designs used the existing single-start damped least-squares solver: 40 iterations, damping of 1 mm, normalized component step limit 0.1, finite-difference step $10^{-4}$ and stopping residual 0.01 mm. Each target started at the same feasible exterior lengths 75/0/0 mm with zero rotations. There were no restarts or target-derived initial guesses. Deployment fractions enforced the measured coupled domain.

Each target was weighted by its swept-cell volume divided by the number of targets in that cell. Regional percentiles were computed over weighted target residuals. Geometric axial symmetry does not make fixed-start solver performance independent of azimuth: zero-rotation initialization can favour some directions. Accordingly, the full-azimuth results must not be directly compared with the draft's earlier canonical-plane success rates.

\subsection{Verification, Support and Regional Definitions}
\label{subsec:numerical_support}
Median maps require at least 30 configurations or targets per cell, as applicable. Local P95 and failure maps require at least 100 targets. The final shared target set provides this support over @@SUPPORT@@\% of the original reference volume, @@SUPPORTCLAIM@@ the predefined 90\% coverage goal. Unsupported cells are masked, while the global target summary retains the available attempts. Coverage is reported explicitly rather than treating blank cells as successful.

The dense vectorised solver agreed with the native viewer solver to within @@SCALAR@@ mm in residual on 12 checked targets per design. Additional sample-count prefixes, 2.5/5/10 mm grid comparisons and an independent spatial repeat were evaluated. Changing the proposed configuration's maximum gap from 108 to 108.5 mm produced identical sampled states: nesting already made the upper gap stop inactive for that design. This does not imply identical sensitivity for every geometry.

Four reference regions were defined:
\begin{enumerate}
\item The forward corridor: cells with $r<10$ mm and $40\leq z<130$ mm.
\item The low-dexterity region: the frozen lowest-20\%-by-volume region from the earlier measured original-tube study, mapped to the current grid.
\item An IK-problem region: cells with at least 100 independent original-only calibration targets and at least 10\% fixed-start failures. This calibration used @@NCAL@@ targets and was separate from the main comparison.
\item The remaining original reference cells outside the union of those regions.
\end{enumerate}
The first three regions may overlap; their volumes and target counts must not be added. The remaining region may include poorly supported boundary cells and is not assumed to be an easy-control region.

Figures~\ref{fig:ik_target_support} and~\ref{fig:numerical_sampling_support} document the support and discretization checks.

@@SUPPORTFIG@@
@@CONVERGENCE@@
\FloatBarrier

\section{Characterisation of the Original Configuration}
\label{sec:original_characterisation}
The archived original-model study reported an occupied workspace of 48,656.99 cm$^3$ on a 10 mm grid. Increasing its training sample from 150,000 to 200,000 configurations changed the estimate by approximately 0.15\%, and an independent sample covered 98.51\% of its reference occupancy. These historical figures are retained as context under their original assumptions.

@@HISTORY@@

The archived 100,000-target evaluation reported 91.96\% volume-weighted success, median residual 0.000378 mm and P95 residual 217.961 mm, with sufficient local P95/failure support over 91.15\% of its reference volume. Small typical residuals coexisted with a large failed-solve tail. These values are not measurements of the installed robot and are not used to calculate improvements in the current study.

The measured-hardware baseline has a different admissible domain, initial state, target convention and grid. Those changes explain why its absolute scores differ substantially from the historical characterisation. The controlled tube comparison below keeps these settings consistent between designs.

\section{Optimisation Method and Configuration Selection}
\label{sec:optimisation_method}
The measured-hardware search considered total lengths, middle/outer curved lengths and precurvatures while holding diameters and nominal Young's modulus fixed and retaining a straight inner element. The original physical tubes remained the reference. An earlier search found no demonstrated candidate that improved every tested metric simultaneously; this motivated explicit trade-off priorities.

The subsequent search prioritised weak-region and forward-corridor fixed-start reaching, supported by lower-tail isotropy. Workspace and maximum reach were secondary, with predeclared allowable losses of 5\%; additional bounds limited losses in reference-cell retention, global isotropy and reaching. The ranking weighted weak and corridor success gains twice as strongly as global success gains. These were engineering benchmark preferences, not clinical requirements.

There were 115 valid screened designs including references, 26 dense spatial assessments including references, and eight candidate path assessments. Two candidates passed the training checks. The first-ranked candidate used total lengths 350/177.5/87.5 mm and precurvatures 0/21.37/14.04 m$^{-1}$. It was frozen before two independent target/path banks. The second candidate was subsequently evaluated on separate fresh banks after the first failed path protection; this sequential extension is retained in the study records.

Neither candidate passed the full held-out path-preservation rule. The first is retained here as a conditional point-to-point trade-off candidate, not as a design satisfying every original acceptance criterion. The dense maps in this chapter characterise that frozen first candidate without further parameter tuning. They use new sampling and therefore do not replace or retrospectively alter the earlier validation results.

\section{Original versus Proposed Configuration}
\label{sec:configuration_comparison}
\subsection{Parameter and Shape Changes}
\label{subsec:parameter_comparison}
\begin{table}[htbp]
\centering\small
\caption{Tube parameters in inner/middle/outer order. Total lengths provisionally mean chuck-front-to-distal-tip material lengths.}
\label{tab:optimised_parameters}
\begin{tabular}{lcc}\toprule
Parameter & Original & Proposed\\\midrule
Total length (mm) & 350/170/80 & 350/177.5/87.5\\
Straight length (mm) & 350/80/15 & 350/87.5/22.5\\
Curved length (mm) & 0/90/65 & 0/90/65\\
Precurvature (m$^{-1}$) & 0/19.12/14.04 & 0/21.37/14.04\\
Outer diameter (mm) & 0.50/0.70/0.90 & 0.50/0.70/0.90\\
Inner diameter (mm) & 0/0.62/0.80 & 0/0.62/0.80\\
Young's modulus (GPa) & 75/75/75 & 75/75/75\\\bottomrule
\end{tabular}
\end{table}
Figure~\ref{fig:optimised_tube_shapes} shows the parameter changes listed in Table~\ref{tab:optimised_parameters}.

@@SHAPES@@

The middle and outer total lengths increase by 7.5 mm, while middle precurvature increases by approximately 11.77\%. Curved lengths are unchanged. The 90 mm middle curve exceeds its maximum exterior exposure of 82.5 mm; the exterior model therefore cannot identify a unique middle curved length above that threshold at fixed total length and curvature. The selected values are not a manufacturing optimum. Internal guide passage, retained material and permissible bending remain unresolved.

@@VIEWER@@

\FloatBarrier
\subsection{Workspace Comparison}
\label{subsec:workspace_comparison}
The primary 5 mm occupancy estimate changes from @@VOL0@@ to @@VOL1@@ cm$^3$, a change of @@VOLGAIN@@\%. The independent repeat gives @@REPVOL0@@ and @@REPVOL1@@ cm$^3$, respectively. These estimates remain sensitive to cell size and sampling density; the proposed occupancy is not asserted to be a strict geometric superset.

Figures~\ref{fig:workspace_comparison} and~\ref{fig:workspace_overlay} show the distribution of shared, gained and original-only occupancy.

@@WORKSPACE@@
@@OVERLAY@@

\FloatBarrier
\subsection{Positional Dexterity Comparison}
\label{subsec:isotropy_comparison}
On common supported original-reference cells, the volume-weighted mean of cell-median isotropy changes from @@ISO0@@ to @@ISO1@@ (@@ISOGAIN@@\%). These cells cover @@ISOSUPPORT@@\% of the original reference volume. Forward-corridor means change from @@CORR0@@ to @@CORR1@@ (@@CORRGAIN@@\%). Common-reference spatial P10 changes from @@P100@@ to @@P101@@. The difference map identifies where gains or losses occur rather than assuming uniform improvement.

Figures~\ref{fig:isotropy_comparison} and~\ref{fig:isotropy_difference} distinguish regional changes from the global average.

@@ISO@@
@@DIFF@@

\FloatBarrier
\subsection{Fixed-Start IK Comparison}
\label{subsec:ik_comparison}
Both designs were evaluated on @@NTARGET@@ identical fresh targets. Volume-weighted success changes from @@SUCCESS0@@\% to @@SUCCESS1@@\%, a difference of @@SUCCESSGAIN@@ percentage points. The global P95 changes from @@P950@@ to @@P951@@ mm. However, the maximum residual changes from @@MAX0@@ to @@MAX1@@ mm; typical and upper-tail summaries must therefore be interpreted alongside failures rather than as a uniform accuracy improvement.

Table~\ref{tab:fixed_start_comparison} gives the global statistics. Figures~\ref{fig:ik_median_comparison}--\ref{fig:ik_failure_comparison} locate typical errors, upper-tail residuals and failures.

@@IKTABLE@@
@@MEDIAN@@
@@P95@@
@@FAILURE@@

\FloatBarrier
\subsection{Regional Comparison}
\label{subsec:regional_comparison}
Regional summaries complement the maps. Isotropy tables use common supported spatial cells; IK statistics use the available weighted target attempts within each frozen region. These denominators differ and their support is reported explicitly.

Tables~\ref{tab:regional_isotropy} and~\ref{tab:regional_ik} separate supported spatial summaries from numerical reaching results. Only 14.83\% of the remaining region has common 30-state isotropy support; its isotropy mean describes that subset and must not be treated as representative of the entire remaining region.

@@ISOTABLE@@
@@REGIK@@

@@REGIONALTEXT@@ Higher positional isotropy does not guarantee successful fixed-start IK, and failed numerical attempts do not prove that the target is outside the geometric workspace. The proposed configuration is consequently evaluated as a task-dependent trade-off.
\FloatBarrier

\section{Path-Following Evaluation and Failure Diagnosis}
\label{sec:path_diagnosis}
The earlier independent path evaluation used two banks of 24 baseline reference trajectories. The original tubes completed 44/48 paths and the proposed tubes completed 43/48 under the original initialization procedure. Completion required all 81 checked deviations along a path to be at most 0.5 mm. The average of the two banks' P95 tracking residuals increased from 0.087 to 0.153 mm; this is an average of percentiles, not a pooled percentile or maximum error.

The one-path reduction was a net change: the proposed tubes completed two paths the original tubes failed, but failed three that the original completed. Two new failures had maximum deviations of 12.05 and 37.84 mm. In the first, the outer tube reached full extension; in the second, it reached the exterior model's zero-exposure boundary. The solver then made little progress while the reference continued.

Increasing the iteration budget or reducing step size alone did not resolve those starts. Independent pointwise solves and reverse continuation identified alternative initial tube configurations at essentially the same tip position. With the same 20 motion steps and 40-iteration controller budget, these starts reduced the maximum deviations to 0.027 and 0.115 mm when checked at 2,001 positions per trajectory. All checked states respected the same measured coupled limits.

Figure~\ref{fig:path_recovery} contrasts the failed trajectories and their diagnostic recoveries.

@@PATH@@

These results demonstrate feasible low-error alternatives within the ideal model and implicate initialization and boundary handling in the observed failures. They do not establish a robust automatic initialization method. The alternative starts were identified after inspecting the failed cases and are reported as diagnostic recovery tests, not substituted into the original 43/48 score. A transition from a prescribed existing robot posture to an alternative start was not evaluated. A fair controller-upgrade study would apply the same automatic method to both geometries and evaluate fresh paths; that study remains future work.

\clearpage
\section{Principal Findings}
\label{sec:numerical_findings}
The measured carriage and clearance constraints substantially restrict the simulated domain and require coupled actuator modelling. Under those same constraints, the proposed tubes yield @@VOLGAIN@@\% greater sampled primary occupancy and a @@SUCCESSGAIN@@-percentage-point change in global fixed-start success in the dense evaluation. Spatial maps and regional summaries show how these aggregate changes are distributed and retain local weaknesses and losses explicitly.

The tube candidate is supported as a potential numerical point-to-point improvement within the declared benchmarks. It is not a universally improved or physically validated replacement. The path investigation identifies avoidable numerical failures in two cases but leaves the original benchmark intact. Sampling dependence, model simplifications, provisional installed lengths and unresolved internal guidance limit the strength of the claims and motivate subsequent controller and experimental validation.
\FloatBarrier
'''
values={'NTARGET':f"{ga['n']:,}",'NCAL':f"{len(np.load(DATA/'calibration_targets.npz')['targets']):,}",'SUPPORT':f'{support:.2f}','SUPPORTCLAIM':'exceeding' if support>=90 else 'falling below','SCALAR':f'{max(v.values()):.2g}',
'VOL0':f"{a['workspace_cm3']:.2f}",'VOL1':f"{b['workspace_cm3']:.2f}",'VOLGAIN':f"{100*(b['workspace_cm3']/a['workspace_cm3']-1):+.2f}",'REPVOL0':f"{checks['original']['repeat_volume_cm3']:.2f}",'REPVOL1':f"{checks['proposed']['repeat_volume_cm3']:.2f}",
'ISO0':f"{sp['global']['original']['mean']:.5f}",'ISO1':f"{sp['global']['proposed']['mean']:.5f}",'ISOGAIN':f"{100*(sp['global']['proposed']['mean']/sp['global']['original']['mean']-1):+.2f}",'ISOSUPPORT':f"{sp['global']['support_fraction']*100:.2f}",'CORR0':f"{sp['corridor']['original']['mean']:.5f}",'CORR1':f"{sp['corridor']['proposed']['mean']:.5f}",'CORRGAIN':f"{100*(sp['corridor']['proposed']['mean']/sp['corridor']['original']['mean']-1):+.2f}",'P100':f"{sp['global']['original']['p10']:.5f}",'P101':f"{sp['global']['proposed']['p10']:.5f}",
'SUCCESS0':f"{ga['success_pct']:.3f}",'SUCCESS1':f"{gb['success_pct']:.3f}",'SUCCESSGAIN':f"{gb['success_pct']-ga['success_pct']:+.3f}",'P950':f"{ga['p95_mm']:.3f}",'P951':f"{gb['p95_mm']:.3f}",'MAX0':f"{ga['max_mm']:.3f}",'MAX1':f"{gb['max_mm']:.3f}"}
values['TOPUP']=f"The initial target bank did not meet the predefined local support goal. A count-only top-up added independent original-model samples over {len(top)} additional batches, without using candidate residuals to choose cells or changing the frozen regions. The initial bank and its results were archived."
parts=[]
for k,label in region_names.items():
 x=a['regions'][k]['success_pct'];y=b['regions'][k]['success_pct'];parts.append(f"{label} success changed from {x:.2f}\\% to {y:.2f}\\% ({y-x:+.2f} percentage points).")
values['REGIONALTEXT']=' '.join(parts)
values.update(figs);values['SUPPORTFIG']=values['SUPPORT'];values['SUPPORT']=f'{support:.2f}' # caption block and numeric use separate keys
# Restore SUPPORTFIG from the saved figure dictionary (SUPPORT is its figure key).
values['SUPPORTFIG']=figs['SUPPORT']
for k,vv in values.items():text=text.replace('@@'+k+'@@',vv)
assert not re.search(r'@@[A-Z0-9]+@@',text)
(PKG/'chapter5.tex').write_text(text)
(PKG/'main.tex').write_text(r'''\documentclass[12pt,a4paper]{report}
\usepackage[margin=25mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx,booktabs,tabularx,array,amsmath,amssymb,placeins}
\usepackage[hidelinks]{hyperref}
\begin{document}
\setcounter{chapter}{4}
\input{chapter5}
\end{document}
''')
(PKG/'README.md').write_text('''# Revised Chapter 5 - Overleaf package

1. Back up your current Overleaf chapter and ch5 figures.
2. Replace the contents of your Chapter 5 with chapter5.tex and upload the ch5 folder. The chapter keeps the main existing section/table/figure labels where applicable.
3. Your preamble needs graphicx, booktabs, tabularx, array, amsmath, amssymb and placeins. main.tex is a standalone review wrapper with Chapter 5 numbering; do not paste its documentclass or document environment into your full dissertation.

The proposed design is 350/177.5/87.5 mm, curves 0/90/65 mm, precurvatures 0/21.37/14.04 per metre. This package uses measured coupled hardware, no grip offset, the existing fixed-start solver, and full-azimuth targets. It is not the older 35 mm-offset study or an improved-controller evaluation.

Historical original-model characterisation remains explicitly separated. The earlier historical simulator screenshot was replaced by a clearly identified current proposed-viewer figure in the comparison subsection. The supplied chapter is archived outside this package. Path recovery remains post-hoc and does not replace the 43/48 benchmark.

Review the prose in your own voice and reconcile surrounding chapters' hardware, parameter and optimisation claims. No physical or clinical validation is claimed. Raw data, protocols, support top-up, and figure sources are saved in the parent study directory.
''')
(OUT/'change_log.md').write_text('''# Changes from supplied Chapter 5

- Replaced obsolete 35 mm offset, exposure caps, old candidate parameters and old controlled-comparison values.
- Added measured distance equations, coupled limits and provisional installed-length/internal-guidance assumptions.
- Preserved the historical original-model characterisation as context, not a control for optimisation gains.
- Added a fresh full-azimuth identical-target evaluation, independent calibration regions, explicit cell support and count-only target top-up.
- Rebuilt intrinsic-shape, occupancy, gained/lost overlay, isotropy, difference, median/P95/failure and support figures.
- Aligned descriptive isotropy tables with 30-sample map support; distinguished them from earlier search gates.
- Documented actual trade-off search and its failed held-out path protections without presenting the candidate as universally optimal.
- Added a separate path diagnosis and retained the original 43/48 benchmark.
- Preserved existing viewers, previous data and the exact supplied LaTeX.
''')
# Static integration checks; compile result documented separately if available.
labels=re.findall(r'\\label\{([^}]+)\}',text);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',text);images=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text)
assert len(labels)==len(set(labels));assert set(refs)<=set(labels);assert all((PKG/p).exists() for p in images)
stack=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',text):
 if m[1]=='begin':stack.append(m[2])
 else:assert stack.pop()==m[2]
assert not stack
report=dict(unique_labels=len(labels),resolved_internal_references=len(refs),existing_figure_files=len(images),balanced_environments=True,latex_compiled=False)
(OUT/'latex_checks.json').write_text(json.dumps(report,indent=2))
with zipfile.ZipFile(OUT/'chapter5_measured_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in PKG.rglob('*'):
  if p.is_file() and p.suffix in {'.tex','.png','.md'}:z.write(p,p.relative_to(PKG))
print('LaTeX and Overleaf package created',report)

# Multi-floor navigation figure

**Suggested caption.** Evidence-guided floor switching. Negative inspections
reduce support for local search candidates. A switch request requires a margin
in the mean residual search score and must satisfy timing constraints; a
credible, untested target on the current floor defers switching. The accepted
request first seeks a geometric stair flight. The dashed connector denotes a
candidate crossing between the two views, while green segments illustrate
approach and continued search. Only a completed transition creates a confirmed
StairEdge. Floor images and the stair guide come from HM3D scene 00873; the
policy state is schematic, not a recorded episode.

The figure shows the flight case. Detected staircase tracks and geometric
portals are fallback cues described in the text; a portal alone does not certify
reachability. No formula or numeric score is repeated in this figure.

Render with `python3 scripts/render_search_floor_decision.py`.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/search_floor_decision.pdf}
  \caption{Evidence-guided floor switching. Negative inspections reduce local
  support, and a score margin can trigger a switch subject to target-presence
  and timing constraints. The accepted request prioritizes a geometric stair
  flight. The dashed connector illustrates a candidate crossing; only a
  completed transition adds a confirmed StairEdge, after which search resumes
  on the destination floor. The policy state is schematic over rendered HM3D
  floor plans.}
  \label{fig:floor_switch}
\end{figure*}
```

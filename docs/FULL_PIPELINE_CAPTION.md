# Full pipeline figure

**Suggested caption.** System-level mapping and navigation pipeline. RGB-D
observations and pose update object tracks and spatial geometry, while visibility
and arrival evidence revise object presence. These outputs update the persistent
3D scene graph, whose building, storey, room, container, and object nodes are
illustrated over two rendered HM3D floor plans. A stair edge joins the two
storey nodes rather than belonging to the containment hierarchy. Given a target
query, the revised graph and map support search-goal selection and multi-floor
navigation. The navigator emits an action; arrival and completed-transition
results return to the persistent graph. The centered building node branches into two storeys, four rooms, and
subsequent container and object layers above the two floor plans. Node counts
and connections are schematic, while the floor-plan images are scene renders.

Render with `python3 scripts/render_full_pipeline.py`. Use the PDF at the top
of the Methodology section:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/full_pipeline.pdf}
  \caption{System-level mapping and navigation pipeline. RGB-D observations
  update object tracks and spatial geometry; presence revision updates the
  persistent 3D scene graph. The graph and target query guide goal selection
  and multi-floor navigation, which produces an action and returns arrival
  and stair-transition evidence. The floor plans are rendered HM3D scenes;
  raised graph nodes are schematic.}
  \label{fig:full_pipeline}
\end{figure*}
```

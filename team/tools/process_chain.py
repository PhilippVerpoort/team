from typing import Optional, Callable

import numpy as np
import pandas as pd
from numpy.linalg import solve

from cet_units import Q, U


# Try to import igraph if installed.
try:
    import igraph
    from igraph import Graph, Layout

    HAS_IGRAPH: bool = True
except ImportError:
    igraph = None

    class Graph:
        pass

    class Layout:
        pass

    HAS_IGRAPH: bool = False


class ProcessChain:
    _proc_graph: dict[str, dict[str, list[str]]]
    _flows: list[str]
    _name: str | None

    @property
    def name(self) -> str | None:
        return self._name

    # Get name of value chain.
    def __init__(self,
                 process_diagram: Optional[str] = None,
                 process_tree: Optional[dict[str, dict[str, list[str]]]] = None,
                 name: str | None = None,
                 ):
        if process_diagram is None and process_tree is None:
            raise Exception('Either the process_diagram or the process_tree '
                            'argument must be provided.')
        if process_diagram is not None and process_tree is not None:
            raise Exception('The process_diagram and process_tree arguments '
                            'cannot both be provided.')

        self._proc_graph = (
            self._read_diagram(process_diagram)
            if process_diagram is not None else
            process_tree
        )
        self._flows = list({
            flow
            for proc_edges in self._proc_graph.values()
            for flow in proc_edges
        })
        self._name = name

    # Get process graph as property.
    @property
    def proc_graph(self) -> dict[str, dict[str, list[str]]]:
        return self._proc_graph

    # Get process graph as igraph object for plotting.
    def igraph(self) -> tuple[Graph, Layout]:
        if not HAS_IGRAPH:
            raise ImportError("Need to install the `igraph` package first. "
                              "Please run `pip install igraph` or `poetry add "
                              "igraph`.")

        procs = list(self._proc_graph.keys())
        graph = igraph.Graph(
            n=len(procs),
            edges=[
                (procs.index(p1), procs.index(p2))
                for p1 in procs
                for flow, procs2 in self._proc_graph[p1].items()
                for p2 in procs2
            ],
        )
        graph.vs['name'] = procs
        graph.es['name'] = [
            flow
            for p1 in procs
            for flow in self._proc_graph[p1]
        ]

        layout = graph.layout_reingold_tilford(root=[len(graph.vs) - 1])
        layout.rotate(angle=90)

        return graph, layout

    # Reduce a single subdiagram.
    @staticmethod
    def _reduce_subdiagram(subdiagram: str) -> tuple[str, str, str]:
        processes = []
        for token in subdiagram.split('=>'):
            components = token.split('->')
            if len(components) == 1:
                processes.append((token.strip(' '), None))
            elif len(components) == 2:
                processes.append((components[0].strip(' '),
                                  components[1].strip(' '),))
            else:
                raise Exception("Too many consecutive `->` in diagram.")

        for i in range(len(processes)):
            proc, flow = processes[i]
            proc2 = processes[i + 1][0] if i + 1 < len(processes) else None
            if flow is None and i + 1 < len(processes):
                raise Exception(f"Flow must be provided for processes feeding "
                                f"into downstream processes: {subdiagram}")
            yield proc, flow, proc2

    # Read the full diagram.
    @staticmethod
    def _read_diagram(diagram: str) -> dict[str, dict[str, list[str]]]:
        out = {}
        for diagram in diagram.split(';'):
            for proc, flow, proc2 in ProcessChain._reduce_subdiagram(diagram):
                if flow is None:
                    continue
                if proc in out:
                    if flow in out[proc]:
                        if proc2 is not None:
                            out[proc][flow].append(proc2)
                    else:
                        out[proc] |= {
                            flow: ([proc2] if proc2 is not None else [])
                        }
                else:
                    out[proc] = {flow: ([proc2] if proc2 is not None else [])}
                if proc2 is not None and proc2 not in out:
                    out[proc2] = {}

        return out

    def calc_scaling(self,
                     data: dict[str, float | Q | pd.Series] | pd.DataFrame,
                     **kwargs):
        # Convert dict to DataFrame if necessary.
        if isinstance(data, dict):
            df = pd.DataFrame(
                data={
                    var_name: (var_values.m
                               if isinstance(var_values, Q) else
                               var_values)
                    for var_name, var_values in data.items()
                },
                index=(
                    [0]
                    if all(not isinstance(c, pd.Series)
                           for c in data.values())
                    else None
                ),
            ).astype({
                var_name: f"pint[{var_values.u:~}]"
                for var_name, var_values in data.items()
                if isinstance(var_values, Q)
            }).astype({
                var_name: "pint[dimensionless]"
                for var_name, var_values in data.items()
                if isinstance(var_values, float)
            })
        else:
            df = data

        # Apply row-wise operation and return.
        return df.apply(self.calc_scaling_single, axis=1, **kwargs)

    # Calculate scaling for single row.
    def calc_scaling_single(self,
            row: pd.Series | dict[str, Q],
            func_unit: dict[str, dict[str, Q]] | None = None,
            constraints: Optional[dict[str, dict[str, Q]]] = None,
            pat: Callable | None = None) -> pd.Series:
        # Set demand and constraints as empty dicts if not provided.
        func_unit = func_unit or {}
        constraints = constraints or {}

        # Raise Exception if neither func_unit nor constraints are set.
        if not func_unit and not constraints:
            raise Exception(
                "Either functional unit or constraints must be set."
            )

        # Set default pattern if not provided.
        if pat is None:
            pat = "Tech|{proc}|{io}|{flow}".format

        # Set patterns for input and output.
        pat_input = pat(proc="{0}", io="Input", flow="{1}").format
        pat_output = pat(proc="{0}", io="Output", flow="{1}").format

        # Convert sparse dict to dense NumPy array: Technosphere Matrix (TSM).
        tsm = np.array([
            [
                + row[pat_output(proc1, flow)].m
                if proc1 == proc2 else
                - row[pat_input(proc2, flow)]
                  .to(row[pat_output(proc1, flow)].u).m
                if proc2 in proc1_flow_targets else
                0.0
                for proc2 in self._proc_graph
            ]
            for proc1 in self._proc_graph
            for flow, proc1_flow_targets in self._proc_graph[proc1].items()
        ])

        # Convert dict to NumPy array: Functional Unit (FU).
        fu = np.array([
            func_unit[proc1][flow]
            .to(row[pat_output(proc1, flow)].u).m
            if proc1 in func_unit and flow in func_unit[proc1] else
            0.0
            for proc1 in self._proc_graph
            for flow, proc1_flow_targets in self._proc_graph[proc1].items()
        ])

        # If constraints are set, expand the TSM and the FU.
        if constraints:
            tsm = np.concatenate([
                # Original TSM.
                tsm,
                # Additional constraints.
                [
                    [
                        row[pat_output(proc1, flow)].m
                        if proc1 == proc2 else
                        0.0
                        for proc2 in self._proc_graph
                    ]
                    for proc1 in constraints
                    for flow in constraints[proc1]
                ]
            ])
            fu = np.concatenate([
                # Original demand.
                fu,
                # Additional constraints.
                [
                    constraints[proc1][flow]
                    .to(row[pat_output(proc1, flow)].u).m
                    for proc1 in constraints
                    for flow in constraints[proc1]
                ]
            ])

        # Calculate scaling from TSM and FU.
        scaling = solve(tsm, fu)

        # Return results (scaling) and inputs (FU and constraints). Add
        # prefix if necessary.
        ret = {}
        ret |= {
            f"Scaling|{proc}": scaling[i] * U('')
            for i, proc in enumerate(list(self._proc_graph.keys()))
        }
        if func_unit:
            ret |= {
                f"Functional Unit|{proc}|{flow}": func_unit[proc][flow]
                for proc in func_unit
                for flow in func_unit[proc]
            }
        if constraints:
            ret |= {
                f"Constraints|{proc}|{flow}": constraints[proc][flow]
                for proc in constraints
                for flow in constraints[proc]
            }

        # Convert dict to Series.
        ret = pd.Series(ret)

        # Add name of process chain if provided.
        if self._name is not None:
            ret.rename(
                lambda c: "|".join([
                    (ts:=c.split("|"))[0],
                    self._name,
                    *ts[1:],
                ]),
                inplace=True,
            )

        return ret

import os
import bisect
import numpy as np

from pyquickbench._benchmark import run_benchmark
from pyquickbench.manual_rank_assign import ManualRankAssign
# from pyquickbench._utils import _convert_benchfile_shape

from rich.text import Text
from rich.style import Style

from textual.app import App
from textual.widgets import Footer, Header, Tree

class ImageCompareCLI(App):
    
    CSS_PATH = "CLI.tcss"

    BINDINGS = [
        # ("d", "toggle_dark", "Toggle dark mode"),
        ("ctrl+Return", "toggle_dark", "Toggle dark mode"),
        ("a", "toggle_select", "Toggle group / compare, restrict / unrestrict"),
    ]
    
    def __init__(self, Workspace_dir):
        super().__init__()
        
        self.messages = []
        
        self.Workspace_dir = Workspace_dir
        
        self.bench_filename = os.path.join(self.Workspace_dir, "bench.npz")
        
        if not os.path.isfile(self.bench_filename):
            raise ValueError(f"Benchmark file {self.bench_filename} not found.")

        self.benchfile_shape, self.all_vals = run_benchmark(
            filename = self.bench_filename  ,
            return_array_descriptor = True  ,
            StopOnExcept = True             ,
        )
        
        self.restrict_idx, self.restrict_shape = ManualRankAssign.default_restrict_values(self.benchfile_shape)
        self.compare_intent  = ManualRankAssign.default_compare_intent(self.benchfile_shape)
        
        self.cur_highlighted_node = None
        
    def build_bench_tree(self):
        
        tree = Tree("Benchmark")
        tree.root.expand()
        
        for i, (key, vals) in enumerate(self.benchfile_shape.items()):
        
            keytree = tree.root.add(label = f"{key} : {self.compare_intent[key]}", data = key)
            
            if hasattr(vals, '__getitem__'): 
                vals_it = vals
            elif hasattr(vals, '__len__'):
                vals_it = range(len(vals))
            else:
                vals_it = range(vals)
                
            for j, val in enumerate(vals_it):
                
                if j in self.restrict_idx[i]:
                    label = f"{val} : selected"
                else:
                    label = f"{val} : unselected"
                
                keytree.add_leaf(label = label, data = (key, i, j))

        return tree
        
    def compose(self):
        """Create child widgets for the app."""
        yield Header()
        yield self.build_bench_tree()
        yield Footer()

    def on_tree_node_highlighted(self, event):
        
        self.cur_highlighted_node = event.node
    
    def action_toggle_select(self):
        
        if self.cur_highlighted_node is not None:
            
            data = self.cur_highlighted_node.data
            
            if isinstance(data, str):
                
                cur_intent = self.compare_intent[data]
                if cur_intent == "group":
                    self.compare_intent[data] = "compare"
                elif cur_intent == "compare":
                    self.compare_intent[data] = "group"
                else:
                    raise ValueError(f"Unknown comare intent {cur_intent}")
                
                label = f"{data} : {self.compare_intent[data]}"
                                    
            elif isinstance(data, tuple):
                
                key, i, j = data
                vals = self.benchfile_shape[key]
                
                if hasattr(vals, '__getitem__'): 
                    vals_it = vals
                elif hasattr(vals, '__len__'):
                    vals_it = range(len(vals))
                else:
                    vals_it = range(vals)
                    
                val = vals_it[j]
                
                idx = bisect.bisect_left(self.restrict_idx[i], j)

                isIn = (idx != self.restrict_shape[i])
                if isIn:
                    isIn = (self.restrict_idx[i][idx] == j)

                if isIn:
                    self.restrict_shape[i] -= 1
                    self.restrict_idx[i].pop(idx)
                    label = f"{val} : unselected"

                else:
                    self.restrict_shape[i] += 1
                    self.restrict_idx[i].insert(idx, j)
                    label = f"{val} : selected"

            else:
                
                raise ValueError(f'Unknown data type : {type(data)}')
            
            self.cur_highlighted_node.set_label(label)

    def action_toggle_dark(self):
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

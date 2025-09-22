import os
import bisect
import numpy as np

from pyquickbench._benchmark import run_benchmark
from pyquickbench.manual_rank_assign import ManualRankAssign
# from pyquickbench._utils import _convert_benchfile_shape

from rich.text import Text
from rich.style import Style

# from textual import getters
from textual.app import App
from textual.widgets import Footer, Header, Tree
from textual.widgets import Input, Label, DirectoryTree, Select
from textual.validation import Number, Function
from textual.containers import Horizontal, Vertical

cycle_intents = {
    "compare" : "group",
    "group" : "compare",
}
intents_label_color = {
    "compare" : "green",
    "group" : "orange1",
}

class ImageCompareCLI(App):
    
    CSS_PATH = "CLI.tcss"

    BINDINGS = [
        # ("d", "toggle_dark", "Toggle dark mode"),
        # ("ctrl+Return", "toggle_dark", "Toggle dark mode"),
        ("left", "toggle_select", "Toggle group / compare, restrict / unrestrict"),
        ("right", "toggle_select", "Toggle group / compare, restrict / unrestrict"),
    ]
    
    def __init__(self, Workspace_dir):
        super().__init__()
        
        self.messages = []
        
        self.Workspace_dir = Workspace_dir

        self.cur_highlighted_node = None
        self.cur_focus_id = None
        
    @classmethod
    def load_benchfile(self, bench_filename):
        return run_benchmark(
            filename = bench_filename       ,
            return_array_descriptor = True  ,
            StopOnExcept = True             ,
        )
        
    def populate_bench_tree(self, tree = None):
        
        if tree is None:
            tree = self.query_one("#bench_tree")
            
        tree.clear()

        for i, (key, vals) in enumerate(self.benchfile_shape.items()):
        
            intent = self.compare_intent[key]
            label = Text(f"{key}")
            label.append_text(Text(f"    {intent}", style = Style(color = intents_label_color[intent])))
        
            keytree = tree.root.add(label = label, data = key)
            
            if hasattr(vals, '__getitem__'): 
                vals_it = vals
            elif hasattr(vals, '__len__'):
                vals_it = range(len(vals))
            else:
                vals_it = range(vals)
                
            for j, val in enumerate(vals_it):

                label = Text(f"{val}")
                
                if j in self.restrict_idx[i]:
                    label.append_text(Text("    selected", style = Style(color = "green")))
                else:
                    label.append_text(Text("    unselected", style = Style(color = "red")))

                keytree.add_leaf(label = label, data = (key, i, j))

    def compose(self):
        """Create child widgets for the app."""

        yield Header()

        filelist = [f for f in os.listdir(self.Workspace_dir) if os.path.isfile(f)]
        if len(filelist) == 0:
            filelist = ["Directory is empty"]
        if "bench.npz" in filelist:
            value = "bench.npz"
        else:
            value = Select.BLANK

        with Horizontal(classes="maxheight"):
            yield Label("Benchmark file")
            yield Select.from_values(filelist, value = value)
        
        with Horizontal(classes="maxheight"):
            yield Label("Number of options in a comparison")
            yield Input(placeholder="Enter a number...", value = "2", validators=[Number(minimum=2)])
            yield Label()
        
        tree = Tree("Benchmark", id="bench_tree")
        tree.show_root = False
        tree.root.expand()
        
        # with Horizontal():
        yield Label("Benchmark options")
        yield Label()
        yield tree

        yield Footer()
        
    def on_select_changed(self, event):

        self.bench_filename = os.path.join(self.Workspace_dir, event.value)
        self.benchfile_shape, self.all_vals = self.load_benchfile(self.bench_filename)
        self.restrict_idx, self.restrict_shape = ManualRankAssign.default_restrict_values(self.benchfile_shape)
        self.compare_intent  = ManualRankAssign.default_compare_intent(self.benchfile_shape)
        
        try:
            self.populate_bench_tree()
        except Exception as exc:
            self.notify(f'{exc}', timeout=60)
        
    def on_tree_node_highlighted(self, event):
        self.cur_highlighted_node = event.node
        
    def action_toggle_select(self):
        
        if self.cur_highlighted_node is not None:
            
            if self.cur_highlighted_node.tree.id == "bench_tree":
                
                data = self.cur_highlighted_node.data
                
                if data is not None:
                    
                    if isinstance(data, str):
                        
                        cur_intent = self.compare_intent[data]
                        new_intent = cycle_intents[cur_intent]
                        self.compare_intent[data] = new_intent
                        
                        label = Text(f"{data}")
                        label.append_text(Text(f"    {new_intent}", style = Style(color = intents_label_color[new_intent])))
                                            
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

                        label = Text(f"{val}")

                        if isIn:
                            self.restrict_shape[i] -= 1
                            self.restrict_idx[i].pop(idx)
                            label.append_text(Text("    unselected", style = Style(color = "red")))

                        else:
                            self.restrict_shape[i] += 1
                            self.restrict_idx[i].insert(idx, j)
                            label.append_text(Text("    selected", style = Style(color = "green")))

                    else:
                        
                        raise ValueError(f'Unknown data type : {type(data)}')
                    
                    self.cur_highlighted_node.set_label(label)

    def action_toggle_dark(self):
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

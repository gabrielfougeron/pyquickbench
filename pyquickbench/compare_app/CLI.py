import os
import bisect
import numpy as np

from rich.text import Text
from rich.style import Style

from textual import on
from textual.app import App
from textual.widgets import Footer, Header, Tree
from textual.widgets import Input, Label, Select, Button, RichLog
from textual.validation import Number, Function
from textual.containers import Horizontal, Vertical, Center

from . import GUI
from pyquickbench._benchmark import run_benchmark
from pyquickbench.manual_rank_assign import ManualRankAssign
# from pyquickbench._utils import _convert_benchfile_shape


cycle_intents = {
    "compare" : "group",
    "group" : "compare",
}

intents_label_color = {
    "compare" : "green",
    "group" : "orange1",
}

bench_default_filename = "bench.npz"
comparison_default_filename = "best_count_k_2.npz"

def load_benchfile(bench_filename):
    return run_benchmark(
        filename = bench_filename       ,
        return_array_descriptor = True  ,
        StopOnExcept = True             ,
    )

class BenchmarkTree(Tree):
    
    BINDINGS = [
        ("left", "toggle_select", "Toggle group / compare, restrict / unrestrict"),
        ("right", "toggle_select", "Toggle group / compare, restrict / unrestrict"),
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_highlighted_node = None

    def on_tree_node_highlighted(self, event):
        self.cur_highlighted_node = event.node
        
    def action_toggle_select(self):

        if self.cur_highlighted_node is not None:
        
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

    def populate_bench_tree(self):

        for i, (key, vals) in enumerate(self.benchfile_shape.items()):
        
            intent = self.compare_intent[key]
            label = Text(f"{key}")
            label.append_text(Text(f"    {intent}", style = Style(color = intents_label_color[intent])))
        
            keytree = self.root.add(label = label, data = key)
            
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

class ImageCompareCLI(App):
    
    CSS_PATH = "CLI.tcss"
    
    BINDINGS = [
        ("c", "start_compare_GUI", "Start comparing!"),
    ]

    
    def __init__(self, Workspace_dir):
        super().__init__()
        
        self.messages = []
        self.Workspace_dir = Workspace_dir
        
    def compose(self):
        """Create child widgets for the app."""

        yield Header()

        filelist = [f for f in os.listdir(self.Workspace_dir) if os.path.isfile(f)]
        if len(filelist) == 0:
            filelist = ["Directory is empty"]
            
        if bench_default_filename in filelist:
            bench_value = bench_default_filename
        else:
            bench_value = Select.BLANK      
                  
        if comparison_default_filename in filelist:
            comparison_value = comparison_default_filename
        else:
            comparison_value = Select.BLANK

        with Horizontal(classes="maxheight"):
            yield Label("Load file", classes="setwidth")
            yield Select.from_values(filelist, value = bench_value, id="bench_file_select")
        
        with Horizontal(classes="maxheight"):
            yield Label("Number of options in a comparison", classes="setwidth")
            yield Input(placeholder="Enter a number...", value = "2", validators=[Number(minimum=2)], id="k_input")
        
        tree = BenchmarkTree("Benchmark", id="bench_tree")
        tree.show_root = False
        tree.root.expand()
        
        with Horizontal():
            yield Label("Comparison options", classes="setwidth")
            yield tree
            
        with Horizontal():
            yield Label("Comparison results", classes="setwidth")
            yield RichLog(id = "results_print")
            
        with Horizontal(classes="maxheight"):
            with Center(classes="setwidth"):
                yield Button("Start comparing!", id = "CompareButton")
            yield Label()

        yield Footer()
        
    def on_select_changed(self, event):
        
        if event.control.id == 'bench_file_select':

            try:
                
                tree = self.query_one("#bench_tree")    
                tree.clear()

                tree.bench_filename = os.path.join(self.Workspace_dir, event.value)
                tree.benchfile_shape, tree.all_vals = load_benchfile(tree.bench_filename)
                tree.restrict_idx, tree.restrict_shape = ManualRankAssign.default_restrict_values(tree.benchfile_shape)
                tree.compare_intent = ManualRankAssign.default_compare_intent(tree.benchfile_shape)
                tree.populate_bench_tree()
                
            except Exception as exc:
                self.notify(f'{exc}', timeout=60)
                
    @on(Button.Pressed, "#CompareButton")
    def action_start_compare_GUI(self):
        
        try:
           
            tree = self.query_one("#bench_tree")  
            k = int(self.query_one("#k_input").value)
            
            restrict_values = ManualRankAssign.build_restrict_values(tree.benchfile_shape, tree.restrict_idx)
            
            self.rank_assign = ManualRankAssign(
                benchfile_shape = tree.benchfile_shape  ,
                all_vals = tree.all_vals                ,
                bench_root = self.Workspace_dir         ,
                k = k                                   ,
                compare_intent = tree.compare_intent    ,
                restrict_values = restrict_values       ,
            )

            img_compare_GUI = GUI.ImageCompareGUI(self.rank_assign)
            img_compare_GUI()
            
            self.print_compare_results()
                      
        except Exception as exc:
            self.notify(f'{exc}', timeout=60)
            # raise exc
            
    def print_compare_results(self):
        
        tree = self.query_one("#bench_tree")  
        k = int(self.query_one("#k_input").value)
        
        res_print = self.query_one("#results_print")  
        res_print.clear()
        # res_print.begin_capture_print(stdout=True, stderr=False)

        n_votes, order, v = self.rank_assign.get_order(compare_intent = tree.compare_intent)

        res_print.write(f"Total number of votes: {self.rank_assign.best_count.sum()}")
        res_print.write(f"Total number of votes in comparison: {n_votes}")
        res_print.write("")

        for i, d in enumerate(order):
            
            rank = v.shape[0]-i
            
            res_print.write(f'{rank = }')
            res_print.write(f'P-L weight = {v[i]}')
            
            for key, val in d.items():
                res_print.write(f"{key}:{val}")
                
            res_print.write("") 
        
        # self.end_capture_print(res_print)

            
    def action_toggle_dark(self):
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

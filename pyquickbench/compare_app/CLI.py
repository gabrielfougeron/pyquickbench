import os
import bisect
import asyncio
import numpy as np

from rich.text import Text
from rich.style import Style

from textual import on, work
from textual.app import App
from textual.widgets import Footer, Header, Tree
from textual.widgets import Input, Label, Select, Button, RichLog
from textual.validation import Number, Function
from textual.containers import Horizontal, Vertical, Center

from . import GUI
from pyquickbench._defaults import *
from pyquickbench._benchmark import run_benchmark
from pyquickbench.manual_rank_assign import ManualRankAssign

cycle_intents = {
    "compare" : "group",
    "group" : "compare",
}

intents_label_color = {
    "compare" : "green",
    "group" : "orange1",
}

bench_default_filename = "bench.npz"
save_default_filename = "bench.npz"

def load_benchfile(bench_filename):
    
    file_bas, file_ext = os.path.splitext(bench_filename)
    
    if file_ext != '.npz':
        raise ValueError("Please provide a *.npz file")

    file_content = np.load(bench_filename, allow_pickle = True)
    
    all_vals = file_content.get('all_vals')
    
    # This should stay consistent with _build_args_shapes
    benchfile_shape = {key:val for (key,val) in file_content.items() if key not in ['all_vals', 'compare_intent', 'best_count', fun_ax_name, repeat_ax_name, out_ax_name]}
    benchfile_shape[fun_ax_name] = all_vals.shape[-3]
    benchfile_shape[repeat_ax_name] = all_vals.shape[-2]
    benchfile_shape[out_ax_name] = all_vals.shape[-1]

    restrict_idx, restrict_shape = ManualRankAssign.default_restrict_values(benchfile_shape)
    
    compare_intent = file_content.get('compare_intent')
    if compare_intent is None:
        compare_intent = ManualRankAssign.default_compare_intent(benchfile_shape)
    else:
        compare_intent = compare_intent.flatten()[0] # dirty hack
    
    best_count = file_content.get('best_count')

    return benchfile_shape, all_vals, best_count, restrict_idx, restrict_shape, compare_intent

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

                    self.app.rank_assign = self.app.build_rank_assign()

                else:
                    
                    raise ValueError(f'Unknown data type : {type(data)}')
                
                self.cur_highlighted_node.set_label(label)
 
                self.app.compare_results_with_label()

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
        ("escape", "quit", "Quit app"),
        ("c", "start_compare_GUI", "Start comparing!"),
    ]

    def __init__(self, Workspace_dir):
        super().__init__()
        
        self.Workspace_dir = Workspace_dir
        self.rank_assign = None
        
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
                  
        if save_default_filename in filelist:
            save_filename_value = save_default_filename
        else:
            save_filename_value = ""

        with Horizontal(classes="maxheight"):
            yield Label("Load file", classes="setwidth")
            yield Select.from_values(filelist, value = bench_value, id="bench_file_select")
        
        with Horizontal(classes="maxheight"):
            yield Label("Save file", classes="setwidth")
            yield Input(placeholder="Filename ...", value = save_filename_value, validators=[Function(lambda x:x.endswith(".npz"))], id="save_filename_input")
            yield Label("", classes="setwidth", id="save_filename_exists")
            
        with Horizontal(classes="maxheight"):
            yield Label("Number of options in a comparison", classes="setwidth")
            yield Input(placeholder="Enter a number ...", value = "2", validators=[Number(minimum=2)], id="k_input")
        
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
            yield Label(id="CompareLabel")

        yield Footer()
        
    def on_select_changed(self, event):
        
        if event.control.id == 'bench_file_select':

            try:
                self.load_bench()
            except Exception as exc:
                self.notify(f'{exc}', timeout=60)
                # raise exc
    
    def on_input_changed(self, event):
        
        if event.control.id == 'save_filename_input':
            lbl = self.query_one("#save_filename_exists")  
            if event.value in os.listdir(self.Workspace_dir):
                lbl.content = Text("Warning: File exists", style = Style(color = "orange3"))
            else:
                lbl.content = ""
                
            try:
                self.rank_assign.best_count_filename = event.value
            except:
                pass
                
        elif event.control.id == 'k_input':
            if not event.validation_result.is_valid:
                return
            
            self.load_bench()

    @on(Button.Pressed, "#CompareButton")
    def action_start_compare_GUI(self):
        
        
        lbl = self.query_one("#CompareLabel")  
        lbl.content = Text("Please compare images in GUI, then press Escape.", style = Style(color = "orange3"))
        self.call_after_refresh(self.lauch_GUI_then_compare)

    def lauch_GUI_then_compare(self):

        self.lauch_GUI()
        self.compare_results_with_label()
        
    def compare_results_with_label(self):
    
        lbl = self.query_one("#CompareLabel")  
        lbl.content = Text("Crunching the latest results. Please wait...", style = Style(color = "orange3"))
        self.call_after_refresh(self.print_compare_results_then_reset_label)

    def print_compare_results_then_reset_label(self):

        try:
            self.print_compare_results()
        except Exception as exc:
            self.notify(f'{exc}', timeout=60)
        lbl = self.query_one("#CompareLabel")  

        lbl.content = ""
        
    def lauch_GUI(self):

        img_compare_GUI = GUI.ImageCompareGUI(self.rank_assign)
        img_compare_GUI()
    
    def load_bench(self):
        
        bench_filename = self.query_one("#bench_file_select").value
        
        tree = self.query_one("#bench_tree")    
        tree.clear()

        try:
            tree.bench_filename = os.path.join(self.Workspace_dir, bench_filename)
        except Exception as exc:
            # self.notify(f'{exc}', timeout=60)
            self.rank_assign = None
            self.compare_results_with_label()
            return
        
        tree.benchfile_shape, tree.all_vals, tree.best_count, tree.restrict_idx, tree.restrict_shape, tree.compare_intent = load_benchfile(tree.bench_filename)
        tree.populate_bench_tree()
        
        self.rank_assign = self.build_rank_assign()
        
        save_filename_input = self.query_one("#save_filename_input")  
        self.rank_assign.best_count_filename = save_filename_input.value
        self.compare_results_with_label()
    
    def build_rank_assign(self):
        
        tree = self.query_one("#bench_tree")  
        k = int(self.query_one("#k_input").value)
        
        restrict_values = ManualRankAssign.build_restrict_values(tree.benchfile_shape, tree.restrict_idx)
        
        rank_assign = ManualRankAssign(
            bench_root = self.Workspace_dir         ,
            benchfile_shape = tree.benchfile_shape  ,
            all_vals = tree.all_vals                ,
            compare_intent = {}                     ,
            restrict_values = restrict_values       ,
            best_count = tree.best_count            ,
            k = k                                   ,
        )
        
        rank_assign.best_count_filename = self.query_one("#save_filename_input").value
        
        return rank_assign

    def print_compare_results(self):
 
        res_print = self.query_one("#results_print")  
        res_print.clear()
        
        if self.rank_assign is None:
            return
        
        tree = self.query_one("#bench_tree")  

        try:
            n_votes, order, v = self.rank_assign.get_order(compare_intent = tree.compare_intent)
        except ValueError as exc:
            res_print.write("Not enough items to compare.")
            # res_print.write(exc)
            
            # res_print.write(f"Total number of votes: {self.rank_assign.best_count.sum()}")
            # res_print.write(f"{self.rank_assign.best_count.shape = }")
            # res_print.write(exc)
            return
    
        res_print.write(f"Total number of votes: {self.rank_assign.best_count.sum()}")
        res_print.write(f"Total number of votes in comparison: {n_votes}")

        if np.all(np.isfinite(v)):
            
            res_print.write("")

            for i, d in enumerate(order):
                
                rank = v.shape[0]-i
                
                res_print.write(f'{rank = }, P-L weight = {v[i]:.3f}')
                
                for key, val in d.items():
                    res_print.write(f"{key}: {val}")
                    
                res_print.write("") 
        else:
            res_print.write("Not enough votes for P-L weights assignemnt. Keep comparing!")

    def action_toggle_dark(self):
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

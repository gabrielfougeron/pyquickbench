import os
import sys
import argparse

from . import CLI

CLI_img_compare_parser = argparse.ArgumentParser(
    description = 'Compare images',
    prog = 'imgs-compare',
)

default_Workspace = './'
CLI_img_compare_parser.add_argument(
    '-d', '--dirname',
    default = default_Workspace,
    dest = 'Workspace_dir',
    help = f'Workspace directory. Defaults to the current directory.',
    metavar = '',
)

def CLI_run(cli_args):

    args = CLI_img_compare_parser.parse_args(cli_args)

    root_list = [
        '',
        os.getcwd(),
    ]

    FoundDir = False
    for root in root_list:

        Workspace_dir = os.path.join(root, args.Workspace_dir)

        if os.path.isdir(Workspace_dir):
            FoundDir = True
            break

    if (FoundDir):

        app = CLI.ImageCompareCLI(
            Workspace_dir = args.Workspace_dir
        )
        app.run()
        
        for mess in app.messages:
            print(mess)
                
    else:

        print(f'Workspace directory {args.Workspace_dir} not found.')

def entrypoint_CLI_run():
    CLI_run(sys.argv[1:])

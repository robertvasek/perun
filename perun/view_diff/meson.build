perun_view_diff_dir = perun_dir / 'view_diff'

perun_view_diff_files = files(
    '__init__.py',
)

py3.install_sources(
    perun_view_diff_files,
    subdir: perun_view_diff_dir
)

subdir('flamegraph')
subdir('datatables')
subdir('report')
subdir('sankey')
subdir('short')

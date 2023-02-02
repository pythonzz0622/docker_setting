require(["codemirror/keymap/sublime", "notebook/js/cell", "base/js/namespace"],
    function (sublime_keymap, cell, IPython) {
        cell.Cell.options_default.cm_config.keyMap = 'sublime';
        var cells = IPython.notebook.get_cells();
        for (var c = 0; c < cells.length; c++) {
            cells[c].code_mirror.setOption('keyMap', 'sublime');
        }
    }
);

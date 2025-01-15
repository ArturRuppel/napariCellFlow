manifest = {
    "name": "napariCellFlow",
    "version": "0.1.0",
    "commands": [
        {
            "id": "napariCellFlow.launch",
            "title": "Launch napariCellFlow",
            "python_name": "napariCellFlow:napariCellFlow",
        }
    ],
    "contributions": {
        "widgets": [
            {
                "id": "napariCellFlow._widget",
                "title": "napariCellFlow",
                "python_name": "napariCellFlow:napariCellFlow",
            },
            {
                "id": "napariCellFlow.preprocessing_widget",
                "title": "Preprocessing",
                "python_name": "napariCellFlow.preprocessing_widget:PreprocessingWidget",
            },
            {
                "id": "napariCellFlow.segmentation_widget",
                "title": "Segmentation",
                "python_name": "napariCellFlow.segmentation_widget:SegmentationWidget",
            },
            {
                "id": "napariCellFlow.tracking_widget",
                "title": "Cell Tracking",
                "python_name": "napariCellFlow.cell_tracking_widget:CellTrackingWidget",
            },
            {
                "id": "napariCellFlow.edge_analysis_widget",
                "title": "Edge Analysis",
                "python_name": "napariCellFlow.edge_analysis_widget:EdgeAnalysisWidget",
            },
        ]
    },
}

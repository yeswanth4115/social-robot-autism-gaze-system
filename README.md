# Eye Gaze Tracker

## Set up the calibration environment (Windows)

The project-local `.venv` is ignored by Git. To recreate it, open this folder
in VS Code, open a PowerShell terminal, and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-calibration.txt
```

To activate the environment in the current terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe 04_calibration.py
```

## Run four-feature calibration

1. Connect the webcam and make sure Windows has granted camera access to VS Code
   or the terminal.
2. From the project folder, activate `.venv` as shown above, or use its Python
   executable directly.
3. Start calibration:

   ```powershell
   python 04_calibration.py
   ```

4. The full-screen window shows a 5-by-10 grid of 50 targets across the screen.
   Look directly at each red target when it appears, keep your head still, and
   move only your eyes. Allow the target to settle and continue looking at it
   while samples are collected.
5. Calibration uses four eye-position features (`left_h`, `left_v`, `right_h`,
   `right_v`). Head-pose features are disabled. The script evaluates models
   against held-out targets, then saves the selected model to `gaze_model.pkl`
   and its settings to `gaze_model_metadata.json`. Press `Q` to stop early; an
   interrupted run does not produce a new valid calibration.

At least 15 targets must have enough stable samples to fit a model. If targets
are rejected, improve the lighting, face the webcam, keep both eyes visible, and
rerun calibration. The bundled face-landmarker model is expected at
`models/face_landmarker.task`.

To use the newly saved calibration, run:

```powershell
python 05_gaze_estimation.py
```

Press `Q` in the gaze-estimation window to quit.

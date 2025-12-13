## Running the Code
1. Execute the script and black window appears
2. Press `o` to upload an image
3. Draw **foreground** and **background** scribbles
4. Press `g` to run Graph Cut
5. Optionally upload ground truth (`t`) to compute IoU
6. Save results `s` or reset `r`
7. Quit with q or ESC

## Mouse Controls
| Action            | Description                      |
| ----------------- | -------------------------------- |
| Left click + drag | Draw scribbles (FG / BG / Erase) |
## Keyboard Controls
### Scribble Modes
| Key | Action                       |
| --- | ---------------------------- |
| `f` | Foreground scribble (yellow) |
| `b` | Background scribble (red)    |
| `e` | Erase scribbles              |

### Segmentation
| Key          | Action                           |
| ------------ | -------------------------------- |
| `g`          | Run / update Graph Cut           |
| `r`          | Reset scribbles and segmentation |
| `s`          | Save segmentation mask (PNG)     |
| `q` or `ESC` | Quit the application             |

## File Handling
| Key | Action                                         |
| --- | ---------------------------------------------- |
| `o` | Upload / open an input image                   |
| `t` | Upload ground-truth mask (for IoU computation) |

## Brush Size
| Key | Action                                |
| --- | ------------------------------------- |
| `k` | Open brush thickness selection dialog |
**Important:**
After selecting the brush thickness, you must close the dialog window for the change to take effect and return to the main application.



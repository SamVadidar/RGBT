# RGB and Thermal Feature-level Sensor Fusion (RGBT)
Remark: Before running the python scripts, make sure you are in the RGBT folder, to avoid relative path issues!
## ToDo List

- [x] Finish literature review of both Object Detection and Fusion 
- [x] Find the conversion Matrix btw. RGB and IR Frame
- [x] Prepare the FLIR Dataset for fusion
- [x] Implement Scaled_Yolov4
- [x] Test the RGB frames from FLIR Dataset on Scaled_Yolov4 (Baseline)
- [x] Ablation Study for the Baseline
- [x] Dive deeper into SPP, PAN, FPN, DCN, SAM, GIOU and other relevant papers for Fusion
- [x] Design the Fusion Network
- [x] Vanilla Fusion Training
- [x] CBAM Fusion Training
- [x] Entropy-based Channel Attention Module (EBAM)
- [x] Entropy-based Spatial Attention Module
- [x] Entropy-based Channel & Spatial Attention Module
- [x] EBAM Fusion Training
- [x] Day & Night Analysis
- [x] Synthetic Fog Analysis


## Studied Networks Performances
| Network        | \multicolumn{4}{c|}{mAP@.5} | Img. Size        | Layer Num.       | Para.            |
|----------------|-----------------------------|------------------|------------------|------------------|
|                | Person                      | Bicycle          | Car              | Overall          |                  |                        |            |
| RGB Baseline   | 39.60\%                     | 50.40\%          | 79.40\%          | 56.60\%          | \hspace{14px}320 | \hspace{18px}539       | 52.5 M     |
| IR Baseline    | 49.60\%                     | 54.90\%          | 84.40\%          | 63.00\%          | \hspace{14px}320 | \hspace{18px}539       | 52.5 M     |
| Vanilla Fusion | 56.90\%                     | 56.70\%          | 82.00\%          | 65.20\%          | \hspace{14px}320 | \hspace{18px}871       | 81.8 M     |
| Fusion + CBAM  | 57.60\%                     | 60.50\%          | 83.60\%          | 67.20\%          | \hspace{14px}320 | \hspace{18px}943       | 82.7 M     |
| Fusion + H\_C  | 62.60\%                     | 65.90\%          | 86.00\%          | 71.50\%          | \hspace{14px}320 | \hspace{18px}913       | 82.7 M     |
| RGBT           | 63.70\%                     | 67.10\%          | 86.40\%          | 72.40\%          | \hspace{14px}320 | \hspace{18px}943       | 82.7 M     |
| CFR\_3         | 74.49\%                     | 57.77\%          | 84.91\%          | 72.39\%          | \hspace{14px}640 | \hspace{2px} Not Given | $> $ 276 M |
| RGBT           | extbf{80.10\%}              | \textbf{76.70\%} | \textbf{91.80\%} | \textbf{82.90\%} | \hspace{14px}640 | \hspace{18px}943       | 82.7 M     |


## Studied Networks Performances in Different Scenarios
| Network  | \multicolumn{4}{c|}{mAP@.5} | Img. Size       |
|----------|-----------------------------|-----------------|
|          | Day                         | Night           | Foggy Day                    | Foggy Night                   |                     |
| RGB      | 62.8\%                      | 52.9\%          | \hspace{12px}29.7\%          | \hspace{14px}25.8\%           | \hspace{14px}320    |
| IR       | 70.3\%                      | 59.3\%          | \hspace{24px}-               | \hspace{26px}-                | \hspace{14px}320    |
| Fusion \ | CBAM                        | 71.4\%          | 70.8\%                       | \hspace{12px}60.6\%           | \hspace{14px}64.4\% | \hspace{14px}320 |
| RGBT     | 74.3\%                      | 72.6\%          | \hspace{12px}62.5\%          | \hspace{14px}68.6\%           | \hspace{14px}320    |
| RGBT     | extbf{84.5\%}               | \textbf{82.2\%} | \hspace{9px} \textbf{78.6\%} | \hspace{11px} \textbf{80.9\%} | \hspace{14px}640    |

import timm
import timm.models.swin_transformer as swin
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes = 0)

for name, module in model.named_modules():
    print(name, "=>", type(module))

print(model.num_features)

optimizer = create_optimizer(model, 'momentum', lr=0.001, weight_decay=0.0001)
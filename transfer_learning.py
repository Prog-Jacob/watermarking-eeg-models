from torch import nn
from enum import Enum


class StudentModelAddedLayers(nn.Module):
    def __init__(self, teacher_model, num_classes=2):
        super(StudentModelAddedLayers, self).__init__()
        self.features = nn.Sequential(*list(teacher_model.children())[:-2])
        self.teacher_lin1 = teacher_model.lin1
        teacher_lin2_in_features = teacher_model.lin2.in_features

        # Adding a new linear layer with appropriate input and output sizes
        self.new_lin1 = nn.Linear(1024, 100)
        self.new_lin2 = nn.Linear(
            100, teacher_lin2_in_features
        )  # Adding another linear layer
        self.dropout = nn.Dropout(0.3)  # Adding dropout for regularization
        self.teacher_lin2 = nn.Linear(teacher_lin2_in_features, num_classes)

        # Freeze the parameters of the pre-existing layers
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.teacher_lin1.parameters():
            param.requires_grad = False

        # Set requires_grad to True for the parameters of the new linear layers
        for param in self.new_lin1.parameters():
            param.requires_grad = True
        for param in self.new_lin2.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.teacher_lin1(x)
        x = self.new_lin1(x)
        x = nn.ReLU(x)  # Applying ReLU activation
        x = self.dropout(x)
        x = self.new_lin2(x)
        x = nn.ReLU(x)  # Applying ReLU activation
        x = self.dropout(x)
        x = self.teacher_lin2(
            x
        )  # Applying the last linear layer from the teacher model
        return x


class StudentModelAllDenseLayers(nn.Module):
    def __init__(self, teacher_model, num_classes=2):
        super(StudentModelAllDenseLayers, self).__init__()
        self.features = nn.Sequential(*list(teacher_model.children())[:-2])
        self.teacher_lin1 = teacher_model.lin1
        teacher_lin2_in_features = teacher_model.lin2.in_features

        # Adding a new linear layer with appropriate input and output sizes
        self.new_lin1 = nn.Linear(1024, 100)
        self.new_lin2 = nn.Linear(
            100, teacher_lin2_in_features
        )  # Adding another linear layer
        self.dropout = nn.Dropout(0.3)  # Adding dropout for regularization
        self.teacher_lin2 = nn.Linear(teacher_lin2_in_features, num_classes)

        # Freeze the parameters of the pre-existing layers
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.teacher_lin1.parameters():
            param.requires_grad = True

        # Set requires_grad to True for the parameters of the new linear layers
        for param in self.new_lin1.parameters():
            param.requires_grad = True
        for param in self.new_lin2.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.teacher_lin1(x)
        x = self.new_lin1(x)
        x = nn.ReLU(x)  # Applying ReLU activation
        x = self.dropout(x)
        x = self.new_lin2(x)
        x = nn.ReLU(x)  # Applying ReLU activation
        x = self.dropout(x)
        x = self.teacher_lin2(
            x
        )  # Applying the last linear layer from the teacher model
        return x


class StudentModelAllLayers(nn.Module):
    def __init__(self, teacher_model, num_classes=2):
        super(StudentModelAllLayers, self).__init__()
        self.features = nn.Sequential(*list(teacher_model.children())[:-2])
        self.teacher_lin1 = teacher_model.lin1
        teacher_lin2_in_features = teacher_model.lin2.in_features

        # Adding a new linear layer with appropriate input and output sizes
        self.new_lin1 = nn.Linear(1024, 100)
        self.new_lin2 = nn.Linear(
            100, teacher_lin2_in_features
        )  # Adding another linear layer
        self.dropout = nn.Dropout(0.3)  # Adding dropout for regularization
        self.teacher_lin2 = nn.Linear(teacher_lin2_in_features, num_classes)

        # Freeze the parameters of the pre-existing layers
        for param in self.features.parameters():
            param.requires_grad = True
        for param in self.teacher_lin1.parameters():
            param.requires_grad = True

        # Set requires_grad to True for the parameters of the new linear layers
        for param in self.new_lin1.parameters():
            param.requires_grad = True
        for param in self.new_lin2.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.teacher_lin1(x)
        x = self.new_lin1(x)
        x = nn.ReLU(x)  # Applying ReLU activation
        x = self.dropout(x)
        x = self.new_lin2(x)
        x = nn.ReLU(x)  # Applying ReLU activation
        x = self.dropout(x)
        x = self.teacher_lin2(
            x
        )  # Applying the last linear layer from the teacher model
        return x


class CCNN(Enum):
    Added = StudentModelAddedLayers
    Dense = StudentModelAllDenseLayers
    All = StudentModelAllLayers


class TSCeption:
    def Added(teacher_model):
        teacher_lin1 = teacher_model.fc[0]
        teacher_lin2 = teacher_model.fc[3]
        teacher_lin1_out_features = teacher_lin1.out_features
        teacher_lin2_in_features = teacher_lin2.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin1_out_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin2_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected (fc) layer
        teacher_model.fc = nn.Sequential(
            teacher_lin1,  # Existing first linear layer (frozen)
            new_lin1,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            new_lin2,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            teacher_lin2,  # Final classification layer (trainable)
        )

        # **Freeze existing layers** (except new ones)
        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze the new linear layers
        for param in teacher_model.fc[1].parameters():
            param.requires_grad = True
        for param in teacher_model.fc[4].parameters():
            param.requires_grad = True

        return (
            teacher_model  # Model is modified in place, but also returned for clarity
        )

    def Dense(teacher_model):
        teacher_lin1 = teacher_model.fc[0]
        teacher_lin2 = teacher_model.fc[3]
        teacher_lin1_out_features = teacher_lin1.out_features
        teacher_lin2_in_features = teacher_lin2.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin1_out_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin2_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected (fc) layer
        teacher_model.fc = nn.Sequential(
            teacher_lin1,  # Existing first linear layer (frozen)
            new_lin1,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            new_lin2,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            teacher_lin2,  # Final classification layer (trainable)
        )

        # **Freeze existing layers** (except new ones)
        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze the new linear layers
        for param in teacher_model.fc[0].parameters():
            param.requires_grad = True
        for param in teacher_model.fc[1].parameters():
            param.requires_grad = True
        for param in teacher_model.fc[4].parameters():
            param.requires_grad = True
        for param in teacher_model.fc[7].parameters():
            param.requires_grad = True

        return (
            teacher_model  # Model is modified in place, but also returned for clarity
        )

    def All(teacher_model):
        teacher_lin1 = teacher_model.fc[0]
        teacher_lin2 = teacher_model.fc[3]
        teacher_lin1_out_features = teacher_lin1.out_features
        teacher_lin2_in_features = teacher_lin2.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin1_out_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin2_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected (fc) layer
        teacher_model.fc = nn.Sequential(
            teacher_lin1,  # Existing first linear layer (frozen)
            new_lin1,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            new_lin2,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            teacher_lin2,  # Final classification layer (trainable)
        )

        # **Freeze existing layers** (except new ones)
        for param in teacher_model.parameters():
            param.requires_grad = True  # Unfreeze everything by default

        return (
            teacher_model  # Model is modified in place, but also returned for clarity
        )


class EEGNet:
    def Added(teacher_model):
        teacher_lin = teacher_model.lin
        teacher_lin_in_features = teacher_lin.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin_in_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected (fc) layer
        teacher_model.lin = nn.Sequential(
            new_lin1,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            new_lin2,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            teacher_lin,  # Final classification layer (trainable)
        )

        # **Freeze existing layers** (except new ones)
        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze the new linear layers
        for param in teacher_model.lin[0].parameters():
            param.requires_grad = True
        for param in teacher_model.lin[3].parameters():
            param.requires_grad = True

        return (
            teacher_model  # Model is modified in place, but also returned for clarity
        )

    def Dense(teacher_model):
        teacher_lin = teacher_model.lin
        teacher_lin_in_features = teacher_lin.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin_in_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected (fc) layer
        teacher_model.lin = nn.Sequential(
            new_lin1,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            new_lin2,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            teacher_lin,  # Final classification layer (trainable)
        )

        # **Freeze existing layers** (except new ones)
        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze the new linear layers
        for param in teacher_model.lin[0].parameters():
            param.requires_grad = True
        for param in teacher_model.lin[3].parameters():
            param.requires_grad = True
        for param in teacher_model.lin[6].parameters():
            param.requires_grad = True

        return (
            teacher_model  # Model is modified in place, but also returned for clarity
        )

    def All(teacher_model):
        teacher_lin = teacher_model.lin
        teacher_lin_in_features = teacher_lin.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin_in_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected (fc) layer
        teacher_model.lin = nn.Sequential(
            new_lin1,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            new_lin2,  # Newly added linear layer (trainable)
            nn.ReLU(),
            dropout,
            teacher_lin,  # Final classification layer (trainable)
        )

        # **Freeze existing layers** (except new ones)
        for param in teacher_model.parameters():
            param.requires_grad = True  # Unfreeze everything by default

        return (
            teacher_model  # Model is modified in place, but also returned for clarity
        )

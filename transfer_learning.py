from torch import nn


class CCNN:
    @staticmethod
    def ADDED(teacher_model):
        teacher_lin1 = teacher_model.lin1[0]
        teacher_lin2 = teacher_model.lin2
        teacher_lin1_out_features = teacher_lin1.out_features
        teacher_lin2_in_features = teacher_lin2.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin1_out_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin2_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected layer
        teacher_model.lin1 = nn.Sequential(
            teacher_lin1,
            new_lin1,
            nn.ReLU(),
            dropout,
            new_lin2,
            nn.ReLU(),
            dropout,
        )

        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze the new linear layers
        for param in teacher_model.lin1[1].parameters():
            param.requires_grad = True
        for param in teacher_model.lin1[4].parameters():
            param.requires_grad = True

        return teacher_model

    @staticmethod
    def DENSE(teacher_model):
        teacher_lin1 = teacher_model.lin1[0]
        teacher_lin2 = teacher_model.lin2
        teacher_lin1_out_features = teacher_lin1.out_features
        teacher_lin2_in_features = teacher_lin2.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin1_out_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin2_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected layer
        teacher_model.lin1 = nn.Sequential(
            teacher_lin1,
            new_lin1,
            nn.ReLU(),
            dropout,
            new_lin2,
            nn.ReLU(),
            dropout,
        )

        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze all linear layers
        for param in teacher_model.lin1[0].parameters():
            param.requires_grad = True
        for param in teacher_model.lin1[1].parameters():
            param.requires_grad = True
        for param in teacher_model.lin1[4].parameters():
            param.requires_grad = True
        for param in teacher_model.lin2.parameters():
            param.requires_grad = True

        return teacher_model

    @staticmethod
    def ALL(teacher_model):
        teacher_lin1 = teacher_model.lin1[0]
        teacher_lin2 = teacher_model.lin2
        teacher_lin1_out_features = teacher_lin1.out_features
        teacher_lin2_in_features = teacher_lin2.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin1_out_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin2_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected layer
        teacher_model.lin1 = nn.Sequential(
            teacher_lin1,
            new_lin1,
            nn.ReLU(),
            dropout,
            new_lin2,
            nn.ReLU(),
            dropout,
        )

        for param in teacher_model.parameters():
            param.requires_grad = True  # Unfreeze everything by default

        return teacher_model


class TSCeption:
    @staticmethod
    def ADDED(teacher_model):
        teacher_lin1 = teacher_model.fc[0]
        teacher_lin2 = teacher_model.fc[3]
        teacher_lin1_out_features = teacher_lin1.out_features
        teacher_lin2_in_features = teacher_lin2.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin1_out_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin2_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected layer
        teacher_model.fc = nn.Sequential(
            teacher_lin1,
            new_lin1,
            nn.ReLU(),
            dropout,
            new_lin2,
            nn.ReLU(),
            dropout,
            teacher_lin2,
        )

        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze the new linear layers
        for param in teacher_model.fc[1].parameters():
            param.requires_grad = True
        for param in teacher_model.fc[4].parameters():
            param.requires_grad = True

        return teacher_model

    @staticmethod
    def DENSE(teacher_model):
        teacher_lin1 = teacher_model.fc[0]
        teacher_lin2 = teacher_model.fc[3]
        teacher_lin1_out_features = teacher_lin1.out_features
        teacher_lin2_in_features = teacher_lin2.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin1_out_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin2_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected layer
        teacher_model.fc = nn.Sequential(
            teacher_lin1,
            new_lin1,
            nn.ReLU(),
            dropout,
            new_lin2,
            nn.ReLU(),
            dropout,
            teacher_lin2,
        )

        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze all linear layers
        for param in teacher_model.fc[0].parameters():
            param.requires_grad = True
        for param in teacher_model.fc[1].parameters():
            param.requires_grad = True
        for param in teacher_model.fc[4].parameters():
            param.requires_grad = True
        for param in teacher_model.fc[7].parameters():
            param.requires_grad = True

        return teacher_model

    @staticmethod
    def ALL(teacher_model):
        teacher_lin1 = teacher_model.fc[0]
        teacher_lin2 = teacher_model.fc[3]
        teacher_lin1_out_features = teacher_lin1.out_features
        teacher_lin2_in_features = teacher_lin2.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin1_out_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin2_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected layer
        teacher_model.fc = nn.Sequential(
            teacher_lin1,
            new_lin1,
            nn.ReLU(),
            dropout,
            new_lin2,
            nn.ReLU(),
            dropout,
            teacher_lin2,
        )

        for param in teacher_model.parameters():
            param.requires_grad = True  # Unfreeze everything by default

        return teacher_model


class EEGNet:
    @staticmethod
    def ADDED(teacher_model):
        teacher_lin = teacher_model.lin
        teacher_lin_in_features = teacher_lin.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin_in_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected layer
        teacher_model.lin = nn.Sequential(
            new_lin1,
            nn.ReLU(),
            dropout,
            new_lin2,
            nn.ReLU(),
            dropout,
            teacher_lin,
        )

        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze the new linear layers
        for param in teacher_model.lin[0].parameters():
            param.requires_grad = True
        for param in teacher_model.lin[3].parameters():
            param.requires_grad = True

        return teacher_model

    @staticmethod
    def DENSE(teacher_model):
        teacher_lin = teacher_model.lin
        teacher_lin_in_features = teacher_lin.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin_in_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected layer
        teacher_model.lin = nn.Sequential(
            new_lin1,
            nn.ReLU(),
            dropout,
            new_lin2,
            nn.ReLU(),
            dropout,
            teacher_lin,
        )

        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze everything by default

        # Unfreeze all linear layers
        for param in teacher_model.lin[0].parameters():
            param.requires_grad = True
        for param in teacher_model.lin[3].parameters():
            param.requires_grad = True
        for param in teacher_model.lin[6].parameters():
            param.requires_grad = True

        return teacher_model

    @staticmethod
    def ALL(teacher_model):
        teacher_lin = teacher_model.lin
        teacher_lin_in_features = teacher_lin.in_features

        # Create new layers
        new_lin1 = nn.Linear(teacher_lin_in_features, 100)
        new_lin2 = nn.Linear(100, teacher_lin_in_features)
        dropout = nn.Dropout(0.3)

        # Construct the new fully connected layer
        teacher_model.lin = nn.Sequential(
            new_lin1,
            nn.ReLU(),
            dropout,
            new_lin2,
            nn.ReLU(),
            dropout,
            teacher_lin,
        )

        for param in teacher_model.parameters():
            param.requires_grad = True  # Unfreeze everything by default

        return teacher_model

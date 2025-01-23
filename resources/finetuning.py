import torch
from utils import plot_loss
from tqdm import tqdm 

def ce_finetune(
        model, 
        optimizer, 
        train_loader, 
        n_epochs=1,
        use_scheduler=False,
        time_step=10):
    
    criterion = torch.nn.CrossEntropyLoss()
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader) * n_epochs
        )


    model.train()
    train_losses = []
    learning_rates = [] if use_scheduler else None
    for epoch in range(n_epochs):
        local_losses = []
        for step, batch in enumerate(tqdm(train_loader)):
            images = batch['image'].to(model.device)
            labels = batch['label'].to(model.device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()

            local_losses.append(loss.item())
            if (step) and (step%time_step==0):
                train_losses.append(sum(local_losses)/len(local_losses))
                if use_scheduler:
                    learning_rates.append(scheduler.get_last_lr()[0])
                plot_loss(train_losses, learning_rates)
                local_losses = []
        plot_loss(train_losses, learning_rates)


def kd_finetune(
    model,
    teacher_model,
    optimizer,
    train_loader,
    n_epochs=1,
    use_scheduler=False,
    time_step=10,
    T=2.0,
    ce_loss_weight=0.5
):
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader)*n_epochs
        )

    model.train()
    teacher_model.eval()
    train_losses = []
    learning_rates = [] if use_scheduler else None
    for epoch in range(n_epochs):
        local_losses = []
        for step, batch in enumerate(tqdm(train_loader)):
            images = batch['image'].to(model.device)
            labels = batch['label'].to(model.device)

            with torch.no_grad():
                teacher_logits = teacher_model(images).logits
            student_logits = model(images).logits
            
            kd_loss = torch.nn.functional.kl_div(
                torch.log_softmax(student_logits / T, dim=1),
                torch.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T ** 2)
            ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')(student_logits, labels)

            loss = ce_loss_weight * ce_loss + (1 - ce_loss_weight) * kd_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if use_scheduler:
                scheduler.step()

            local_losses.append(loss.item())
            if (step) and (step % time_step == 0):
                train_losses.append(sum(local_losses) / len(local_losses))
                if use_scheduler:
                    learning_rates.append(scheduler.get_last_lr()[0])
                plot_loss(train_losses, learning_rates)
                local_losses = []
        
        plot_loss(train_losses, learning_rates)

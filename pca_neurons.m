clearvars
close all

load('SpikeTrainsXTrial_1')
behav = load('150628_Behavior.txt');
cellID = load('150628_CellType.txt');

%bin_size = 5; % ms, comment out for TCA
w = gausswin(20); % generate a gaussian filter of length L (this value is a multiple of bin_size)
spks = cell(length(TrData),1);
conv_spks = cell(length(TrData),1);
for j = 1:length(TrData)

    n=1000;
    T = behav(j,1:2);
%     bin_size = (T(2)-T(1))/(1000-1); %Uncomment for TCA
    bin_size = (T(2)-T(1))/(50-1);
    %making bins of equal size from start of trial to end of trial with 5
    %(or whatever bin_size is)distance between them
    bins = T(1)+bin_size/2:bin_size:T(2)-bin_size/2;
    spks{j} = zeros(length(cellID),length(bins)-1);
    % identify the cell (neuron) ID in current trial (j)
    for cell = 1:length(cellID)
        % find index of each cell
        cel_idx = find(TrData{j}(:,1)==cell);
        % select timestamps for this trial and this cell
        ts = TrData{j}(cel_idx,2);
        
        spks{j}(cell,:) = f_timeseries_to_spiketrain(ts,bin_size,T);
        conv_spks{j}(cell,:) = filter(w,1,spks{j}(cell,:));
    end
    
end

%visualize one example trial
j=1;
figure('Units', 'centimeters', 'PaperPositionMode', 'auto','Position',[5 5 22 5]) 
% subplot(1,3,1)
imagesc(spks{j}); hold on
colormap(1-gray)
title('Spike train', 'Fontsize',22)
ylabel('Neuron', 'Fontsize',22)
xlabel('Time bins (ms)','Fontsize',22)
set(gca,'Fontsize',18)
colorbar()
subplot(1,3,2)
imagesc(conv_spks{j}); hold on
title('Convolved spike train')
colorbar()
xlabel('Time bins')
subplot(1,3,3)
for cell = 1:length(cellID)
    if cell == 1
        plot(conv_spks{j}(cell,:),'k-'); hold on
    else
        plot(conv_spks{j}(cell,:)+sum(max(conv_spks{j}(1:cell-1,:),[],2)),'k-'); hold on
    end
end
title({'Convolved spike trains','(as in Williams et al.)'})
title({'Convolved spike trains','Trial-concatenated PCA'})
ylabel('Neuron')
xlabel('Time bins')


%% PCA
pca_spks = cat(2,conv_spks{:});
pca_spks=pca_spks';
pca_spks = normalize(pca_spks);

mean(pca_spks);
var(pca_spks);

[coeff,score,latent,~,explained] = pca(pca_spks); % X=W*A

% score = score./max(score);

bin=length(bins)-1;
cells = zeros(bin,j,5);

for k= 1:5
    initial=0;

    for i = 1:j
        cells(:,i,k) = score(1+initial:bin*i,k);
        initial =initial+bin;
    end
end

cells = cells./max(cells);

 
for fig = 1:5
    figure(fig)
    imagesc(cells(:,:,fig)');
    colormap Jet
    xlabel('Time','Fontsize',22)
    ylabel('Trial','Fontsize',22)
    txt= ['PC', num2str(fig)];
    title(txt, 'Fontsize',22)
    set(gca,'FontSize',18)

end

%% PC AVERAGED PLOTS

for i = 1:5
    correct_indx=find(behav(:,4) == 1);
    incorrect_indx = find(behav(:,4) == 0);
    add_correct =zeros(bin,1);
    for indx = 1:length(correct_indx)
        correct_trials = cells(:,correct_indx(indx),i);
        add_correct = add_correct +correct_trials;
    end
    add_correct=add_correct./length(correct_indx);
    add_incorrect =zeros(bin,1);
    for indx = 1:length(incorrect_indx)
        incorrect_cells = cells(:,incorrect_indx(indx),i);
        add_incorrect = add_incorrect+ incorrect_cells;
    end
    add_incorrect=add_incorrect./length(incorrect_indx);
    outcome = cat(2, add_correct,add_incorrect);
   
    %light position: left =1, right = 0
    left_light_indx=find(behav(:,6) == 1);
    right_light_indx = find(behav(:,6) == 0);
    add_left_light =zeros(bin,1);
    for indx = 1:length(left_light_indx)
        left_light_trials = cells(:,left_light_indx(indx),i);
        add_left_light = add_left_light +left_light_trials;
    end
    add_left_light=add_left_light./length(left_light_indx);
    add_right_light =zeros(bin,1);
    for indx = 1:length(right_light_indx)
        right_light_trials = cells(:,right_light_indx(indx),i);
        add_right_light = add_right_light+ right_light_trials;
    end
    add_right_light=add_right_light./length(right_light_indx);
    light_pos = cat(2, add_left_light,add_right_light);

    %follows light, goes towards light arm: light arm = 1, dark arm =0
    for r = 1:length(behav)
        if behav(r,5) == behav(r,6)
            behav(r,7) = 1;
        else 
            behav(r,7) =0;
        end
    end
 
    light_arm_indx=find(behav(:,7) == 1);
    dark_arm_indx = find(behav(:,7) == 0);
    add_light =zeros(bin,1);
    for indx = 1:length(light_arm_indx)
        light_trials = cells(:,light_arm_indx(indx),i);
        add_light = add_light +light_trials;
    end
    add_light=add_light./length(light_arm_indx);
    add_dark =zeros(bin,1);
    for indx = 1:length(dark_arm_indx)
        dark_trials = cells(:,dark_arm_indx(indx),i);
        add_dark = add_dark+ dark_trials;
    end
    add_dark=add_dark./length(dark_arm_indx);
    following = cat(2, add_light,add_dark);
   
    figure(i)
    subplot(3,1,1)
    imagesc(outcome');
    tx= ['PC',num2str(i), ' Outcome'];
    title(tx,'FontSize',17)
    set(gca,'FontSize',16)
    subplot(3,1,2)
    imagesc(light_pos');
    set(gca,'FontSize',16)
    title('PC2 Light position','FontSize',18)
    subplot(3,1,3)
    imagesc(following');
    title('PC2 Following light','FontSize',16)
    xlabel('Time (ms)')

    txt = ['Session 150628, PC',num2str(i)];
    set(gca,'FontSize',16)

end
% Average over time bins
for i = 1:5
    %average over time bins 
    time_average = mean(cells(:,:,i),1);    
    figure(i)
    imagesc(time_average)
    txt= ['PC', num2str(i), ' Average over Time'];
    title(txt, 'Fontsize',16)
    xlabel('Trials','Fontsize',16)
    set(gca,'FontSize',16)
end

% correlation between time 10-20 to 38-48
PC3_1 = mean(cells(20:25,:,3));
PC3_2 = mean(cells(43:48,:,3));
% 
R = corrcoef(PC3_1,PC3_2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program calculates ENF detection accuracies of 
%
% 1. Huber-LRT - Median Threshold
% 2. LAD-LRT   - Median Threshold
% 3. LS-LRT    - Median Threshold
% 4. LS-LRT
% 5. naive-LRT
%
% versus recording length using real-world audio recordings.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;clear;close all;

%%% Bandpass Filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

F2 = [0 0.4 0.499 0.4995 0.5 0.5005 0.501 0.6 0.8 1];
M2 = [0 0 0 0.2 1 0. 0 0 0 0];
BPF= fir2(1023,F2,M2);
BPFF     = abs(fft(BPF,8192));
scalar   = max(BPFF);
BPF      = BPF/scalar;

%%% Initialization Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fs              = 400;
T               = 1/fs;
AWindowLength   = 16*fs;
AWindowShift    = rectwin(AWindowLength)';
AStepSize       = 1*fs;
NFFT            = 200*fs;

%%%%% Setting for Figure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
duration = 5:10; 
load Threshold_info_5_1_10
thre2 = mean20+2*sqrt(var20);
thre3 = mean30+2*sqrt(var30);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
path         = 'C:\Users\30694\Documents\PHD\Codes\ENF_Detection\Datasets\Recordings\';
H0_index     = dir(strcat(path,'H0'));
H1_index     = dir(strcat(path,'H1'));
ground_truth = [ones(1,length(H1_index)-2),zeros(1,length(H0_index)-2)];

result1      = zeros(length(duration),(length(H1_index)-2+length(H0_index)-2));  % Huber-LRT - Median
result2      = zeros(length(duration),(length(H1_index)-2+length(H0_index)-2));  % LAD-LRT - Median
result3      = zeros(length(duration),(length(H1_index)-2+length(H0_index)-2));  % LS-LRT - Median
result4      = zeros(length(duration),(length(H1_index)-2+length(H0_index)-2));  % LS-LRT
result5      = zeros(length(duration),(length(H1_index)-2+length(H0_index)-2));  % naive-LRT

ACC1         = zeros(1,length(duration));
ACC2         = zeros(1,length(duration));
ACC3         = zeros(1,length(duration));
ACC4         = zeros(1,length(duration));
ACC5         = zeros(1,length(duration));

O_TP1 = zeros(1,length(duration));
O_TN1 = zeros(1,length(duration));
O_FP1 = zeros(1,length(duration));
O_FN1 = zeros(1,length(duration));
O_TP2 = zeros(1,length(duration));
O_TN2 = zeros(1,length(duration));
O_FP2 = zeros(1,length(duration));
O_FN2 = zeros(1,length(duration));
O_TP3 = zeros(1,length(duration));
O_TN3 = zeros(1,length(duration));
O_FP3 = zeros(1,length(duration));
O_FN3 = zeros(1,length(duration));
O_TP4 = zeros(1,length(duration));
O_TN4 = zeros(1,length(duration));
O_FP4 = zeros(1,length(duration));
O_FN4 = zeros(1,length(duration));
O_TP5 = zeros(1,length(duration));
O_TN5 = zeros(1,length(duration));
O_FP5 = zeros(1,length(duration));
O_FN5 = zeros(1,length(duration));

for i = 1:(length(H1_index)-2+length(H0_index)-2)
    disp(['i=',num2str(i)]); 
    if i<=(length(H1_index)-2)
        [audio, fs0] = audioread(strcat(H1_index(i+2).folder,'\',H1_index(i+2).name));
        audio        = audio(:,1)';
    else
        [audio, fs0] = audioread(strcat(H0_index((i-(length(H1_index)-2))+2).folder,'\',H0_index((i-(length(H1_index)-2))+2).name));
        audio        = audio(:,1)';
    end
    
    for j = 1:length(duration)
        current_dur  = duration(j);
        start_index  = randi(length(audio)-current_dur*fs0);
        audio_cut    = audio(start_index:(start_index+current_dur*fs0-1));
        x            = resample(audio_cut, fs, fs0); % Downsampling
        N            = length(x); 
        
        x_filtered   = filter(BPF,1,x); % Bandpass Filtering
        
        NFFT_full    = max(2^18,2^(nextpow2(N)+2));
        X_filtered   = abs(fft(x_filtered,NFFT_full));
        X_filtered   = X_filtered(1:(end/2+1));
        
        % Initial fc
        fc = find(X_filtered==max(X_filtered))*(fs/NFFT_full);

        % Non-Linear LAD
        convergence_threshold_lad = 1e-4; % Set it properly
        iteration = 0;
        CONT=1;
        while CONT 
            iteration = iteration + 1;
            if iteration == 1
                fcc_lad = fc;
            end
            % Optimization wrt theta
            Hm_lad    = [cos(2*pi*T*fcc_lad*(0:N-1))',sin(2*pi*T*fcc_lad*(0:N-1))'];
            theta_lad = ladreg(x_filtered', Hm_lad, false, [], 1); 
            % Optimization wrt fm
            zmin=1e7;
            for m = 1:99
                fm_lad  = fcc_lad + (m-50)*fs/(60*N);  
                Hm_lad  = [cos(2*pi*T*fm_lad*(0:N-1))',sin(2*pi*T*fm_lad*(0:N-1))'];
                zm_lad  = norm(x_filtered' - Hm_lad*theta_lad,1);
                if zm_lad < zmin
                    fcc_new_lad=fm_lad;
                    zmin=zm_lad; 
                end   
            end
            relative_difference_lad = abs(fcc_lad - fcc_new_lad)/fcc_lad ;
            if relative_difference_lad > convergence_threshold_lad
                fcc_lad=fcc_new_lad;
            else
                f_star_lad=fcc_lad;
                CONT=0;
            end
        end

        % Non-Linear Huber Regression 
		c = 1.3415;
		csq = c^2; 
		qn = chi2cdf(csq,1);
		alpha = chi2cdf(csq,3)+csq*(1-qn); % consistency factor for scale 
		
        convergence_threshold = 1e-2; % Set it properly
        iteration = 0;
        CONT=1;
        while CONT 
            iteration = iteration + 1;
            if iteration == 1
                fcc = fc;
            end
            % Optimization wrt theta
            Hm    = [cos(2*pi*T*fcc*(0:N-1))',sin(2*pi*T*fcc*(0:N-1))'];
            [theta, sigma, ~] = hubreg(x_filtered', Hm, c); 
            % Optimization wrt fm
            min_zm = inf;
            for m = 1:99
                fm   = fcc + (m-50)*fs/(60*N);  
                Hm   = [cos(2*pi*T*fm*(0:N-1))',sin(2*pi*T*fm*(0:N-1))'];
                term = (x_filtered' - Hm*theta) / sigma;
			    huber_term = (term .* (abs(term) <= c) + c * sign(term) .* (abs(term) > c)) * sigma;
                zm   = (2 * N / 2) * (alpha * sigma) + huber_term;
                if min(zm) < min_zm
                    min_zm = min(zm);
                    fcc_new= fm;
                end   
            end
            relative_difference = abs(fcc - fcc_new)/fcc ;
            if relative_difference > convergence_threshold
                fcc=fcc_new;
            else
                f_star=fcc;
                CONT=0;
            end
        end
        
        [theta_star,~,~]= hubreg(x_filtered', [cos(2*pi*T*f_star*(0:N-1))',sin(2*pi*T*f_star*(0:N-1))'],c);
        
        Hc1             = [cos(2*pi*T*f_star*(0:N-1))',sin(2*pi*T*f_star*(0:N-1))']; 
        Test_Statistic1 = (x_filtered*Hc1*theta_star)/((norm(x_filtered).^2)); % Huber - Median
        p1(j,i)         = Test_Statistic1;

        theta_star_lad  = ladreg(x_filtered', [cos(2*pi*T*f_star_lad*(0:N-1))',sin(2*pi*T*f_star_lad*(0:N-1))'],false,[],1);
        Hc2             = [cos(2*pi*T*f_star_lad*(0:N-1))',sin(2*pi*T*f_star_lad*(0:N-1))']; 
        Test_Statistic2 = (x_filtered*Hc2*theta_star_lad)/((norm(x_filtered).^2)); % LAD-LRT - Median
        p2(j,i)         = Test_Statistic2;
        
        Hc3              = [cos(2*pi*T*fc*(0:N-1))',sin(2*pi*T*fc*(0:N-1))'];
        Test_Statistic3  = 2/N*(x_filtered*Hc3)*(Hc3'*x_filtered')/((norm(x_filtered).^2)); % LS-LRT - Median
        p3(j,i)          = Test_Statistic3;
        
        Hc4              = [cos(2*pi*T*fc*(0:N-1))',sin(2*pi*T*fc*(0:N-1))'];
        Test_Statistic4  = 2/N*(x_filtered*Hc4)*(Hc4'*x_filtered')/((norm(x_filtered).^2)); % LS-LRT

        Hc5              =[cos(2*pi*T*100*(0:N-1))',sin(2*pi*T*100*(0:N-1))'];
        Test_Statistic5  = 2/N*(x_filtered*Hc5)*(Hc5'*x_filtered')/((norm(x_filtered).^2)); % naive-LRT
        
        if Test_Statistic4 >= thre2(j)
            result4(j,i) = 1; 
        end
         if Test_Statistic5 >= thre3(j)
            result5(j,i) = 1;
        end 
    end
end

thre1m = median(p1,2); 
thre2m = median(p2,2); 
thre3m = median(p3,2);

for j = 1:length(duration)
  for i = 1:(length(H1_index)-2+length(H0_index)-2)    
     if (p1(j,i) >= thre1m(j))
        result1(j,i) = 1;
     end
     if (p2(j,i) >= thre2m(j))
        result2(j,i) = 1;
     end   
     if (p3(j,i) >= thre3m(j))
        result3(j,i) = 1;
     end 
  end
end

for j = 1:length(duration)
    [O_TP1(j),O_TN1(j),O_FP1(j),O_FN1(j)] = fun_TP_TN_FP_FN(result1(j,:),ground_truth);
    ACC1(j)              = (O_TP1(j)+O_TN1(j))/(length(H1_index)-2+length(H0_index)-2);
    [O_TP2(j),O_TN2(j),O_FP2(j),O_FN2(j)] = fun_TP_TN_FP_FN(result2(j,:),ground_truth);
    ACC2(j)              = (O_TP2(j)+O_TN2(j))/(length(H1_index)-2+length(H0_index)-2);
    [O_TP3(j),O_TN3(j),O_FP3(j),O_FN3(j)] = fun_TP_TN_FP_FN(result3(j,:),ground_truth);
    ACC3(j)              = (O_TP3(j)+O_TN3(j))/(length(H1_index)-2+length(H0_index)-2);
    [O_TP4(j),O_TN4(j),O_FP4(j),O_FN4(j)] = fun_TP_TN_FP_FN(result4(j,:),ground_truth);
    ACC4(j)              = (O_TP4(j)+O_TN4(j))/(length(H1_index)-2+length(H0_index)-2);
    [O_TP5(j),O_TN5(j),O_FP5(j),O_FN5(j)] = fun_TP_TN_FP_FN(result5(j,:),ground_truth);
    ACC5(j)              = (O_TP5(j)+O_TN5(j))/(length(H1_index)-2+length(H0_index)-2);
end

%% Calculate Accuracy

figure(1);
pf=plot(duration,ACC1*100,'mo-',duration,ACC2*100,'bx:',duration,ACC3*100,'g*:',duration,ACC4*100,'r+-.', duration, ACC5*100,'k--square' );
pf(1).LineWidth=2;
pf(2).LineWidth=2;
pf(3).LineWidth=2;
pf(4).LineWidth=2;
pf(5).LineWidth=2;

grid on;
hl = legend('Huber-LRT-Median','LAD-LRT-Median','LS-LRT-Median','LS-LRT', 'naive-LRT');
hx = xlabel('$N/f_{\rm{S}}$ (sec)');
hy = ylabel('Accuracy ($\%$)');
set(hx, 'Interpreter', 'latex');
set(hy, 'Interpreter', 'latex');
set(hl, 'Interpreter', 'latex');
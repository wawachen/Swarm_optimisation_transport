clear;
load data_3.mat

figure;
start_point = 1;
end_point = 1300;
plot(posx0(start_point:end_point,3),posy0(start_point:end_point,3),'Linewidth',2)
hold on 
plot(posx1(start_point:end_point,3),posy1(start_point:end_point,3),'Linewidth',2)
hold on 
plot(posx2(start_point:end_point,3),posy2(start_point:end_point,3),'Linewidth',2)

plot(posx0(end_point,3),posy0(end_point,3),'p','MarkerSize',10,'MarkerFaceColor',[1.0,0.0,0.0],'MarkerEdgeColor','k')
plot(posx1(end_point,3),posy1(end_point,3),'p','MarkerSize',10,'MarkerFaceColor',[1.0,0.0,0.0],'MarkerEdgeColor','k')
plot(posx2(end_point,3),posy2(end_point,3),'p','MarkerSize',10,'MarkerFaceColor',[1.0,0.0,0.0],'MarkerEdgeColor','k')

rectangle('Position',[2.8-0.5,0-0.5,1,1],'Curvature',[1,1]),axis equal;
rectangle('Position',[0-0.5,0-0.5,1,1],'Curvature',[1,1]),axis equal;
rectangle('Position',[-2.8-0.5,0-0.5,1,1],'Curvature',[1,1]),axis equal;

set(gca,'linewidth',1.8);
set(gca, 'FontSize',15)
set(gca,'FontWeight','bold');
legend("UAV1","UAV2","UAV3")
xlabel("x (m)")
ylabel("y (m)")
xlim([-5,5])
ylim([-5,5])
xticks([-5:1:5])
yticks([-5:1:5])

clear
load data_4.mat

clear
load data_6.mat

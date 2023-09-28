figure
plot((1:2430)*0.05, lox,(1:2430)*0.05,loy,(1:2430)*0.05,loz)
xlabel("t [s]")
ylabel("orientation [rad]")
title("payload orientation")
legend("roll","picth","yaw")
set(gca,'linewidth',1.2);
set(gca, 'FontSize',10)
set(gca,'FontWeight','bold');
% År
years=[2019, 2020, 2021, 2022, 2023, 2024];
% Antal stängda dagar
dagar = [188, 199, 172, 198, 173, 204];
%procentökning från 35,38 till 54,875 på 8 år
r=(54.875/35.38)^(1/8);
% Kostnad per dag (positivt uttryckt), samt anpassad efter ökad elskatt med
% 5,64% per år
kostnad_per_dag=[20000.*r^2, 20000.*r^3, 20000.*r^4, 20000.*r^5,20000.*r^6, 20000.*r^7];
% Total kostnad per år
total_kostnad = dagar.*kostnad_per_dag;
% Ackumulerad kostnad
Ackumulerad_kostnad =cumsum(total_kostnad);
% Plot
figure;
plot(years, Ackumulerad_kostnad/1e6, '-o', 'LineWidth', 2,'MarkerSize',6);
grid on;
xlabel('År');
ylabel('Ackumulerad kostnad (Mkr)');
%%title('Ackumulerad kostnad för stängningar av råvattenintag (2019–2024)');
% Tvinga x-axeln att visa endast de specifika årtalen
xticks(years);
xticklabels(string(years));  % Konverterar till sträng för säker visning

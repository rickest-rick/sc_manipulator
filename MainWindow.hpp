#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <vector>
#include <algorithm>

#include <QMainWindow>
#include <QFileDialog>
#include <QPushButton>
#include <QBoxLayout>
#include <QSpinBox>
#include <QLabel>
#include <QGroupBox>
#include <QStatusBar>
#include <QMessageBox>

#include "ImageReader.hpp"
#include "QtOpencvCore.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "SeamFunctions.hpp"


class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    
    /* Konstruktor */
    explicit MainWindow(QWidget *parent = 0);
    
    /* Destruktor */
    ~MainWindow();
    
private slots:  
    
    /* Funktionen werden ausgeloest, wenn auf den entsprechenden Button geklickt wird */
    void on_pbOpenImage_clicked();
    void on_pbComputeSeams_clicked();
    void on_pbRemoveSeams_clicked();
    
private:

    
    /* UI Komponenten */
    QWidget     *centralWidget;
    
    QHBoxLayout *horizontalLayout;
    QHBoxLayout *horizontalLayout_2;
    QHBoxLayout *horizontalLayout_3;
    QVBoxLayout *verticalLayout;
    QVBoxLayout *verticalLayout_3;
    
    QPushButton *pbOpenImage;
    QPushButton *pbRemoveSeams;
    QPushButton *pbComputeSeams;
    
    QLabel      *lCaption;
    QLabel      *lCols;
    QLabel      *lRows;
    
    QSpinBox    *sbCols;
    QSpinBox    *sbRows;
    
    QSpacerItem *verticalSpacer;
    QSpacerItem *horizontalSpacer;
    /*****************************************/
    
    /* Originalbild */
    cv::Mat         originalImage;

    /* computed seams */
    std::vector<std::vector<int>> seamsHorizontal;
    std::vector<std::vector<int>> seamsVertical;
    
    /* Methode initialisiert die UI */
    void setupUi();
    
    /* Methoden aktivieren bzw. deaktivieren die UI */
    void enableGUI();
    void disableGUI();

    /* Method that shows error message that no seams are present that can be removed. */
    void noSeamsError();

    /* Method that shows error message that no more non-blocking horizontal seams can be computed. */
    void seamsHorizontalBlockError(int i);

    /* Method that shows error message that no more non-blocking vertical seams can be computed, */
    void seamsVerticalBlockError(int i);
};

#endif // MAINWINDOW_HPP

#include "MainWindow.hpp"
#include <string>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    /* Initialisiere die UI Komponenten */
    setupUi();
}

MainWindow::~MainWindow()
{
    /* loesche die UI Komponenten */
    delete centralWidget;    
    
    /* schliesse alle offenen Fenster */
    cv::destroyAllWindows();
}

/* Methode oeffnet ein Bild und zeigt es in einem separaten Fenster an */
void MainWindow::on_pbOpenImage_clicked()
{
    /* oeffne Bild mit Hilfe eines Dateidialogs */
    QString imagePath = QFileDialog::getOpenFileName(this, "Open Image...", QString(), QString("Images *.png *.jpg *.tiff *.tif"));
    
    /* wenn ein gueltiger Dateipfad angegeben worden ist... */
    if(!imagePath.isNull() && !imagePath.isEmpty())
    {
        /* ...lese das Bild ein */
        cv::Mat img = ImageReader::readImage(QtOpencvCore::qstr2str(imagePath));
        
        /* wenn das Bild erfolgreich eingelesen worden ist... */
        if(!img.empty())
        {
            /* ...merke das Originalbild... */
            originalImage = img;
            
            /* ...aktiviere das UI... */
            enableGUI();
            
            /* ...zeige das Originalbild in einem separaten Fenster an */
            cv::imshow("Original Image", originalImage); 
        }
        else
        {
            /* ...sonst deaktiviere das UI */
            disableGUI();
        }
    }
}

void MainWindow::on_pbComputeSeams_clicked()
{
    /* reset seams */
    seamsVertical.clear();
    seamsHorizontal.clear();

    /* Anzahl der Spalten, die entfernt werden sollen */
    int colsToRemove = sbCols->value();
    
    /* Anzahl der Zeilen, die entfernt werden sollen */
    int rowsToRemove = sbRows->value();
    
    /* Compute energy function. */
    cv::Mat grayscaleImage;
    cv::cvtColor(originalImage, grayscaleImage, cv::COLOR_BGR2GRAY);
    cv::Mat gradientImage;
    seam::sobel(grayscaleImage, gradientImage);
    grayscaleImage.release();

    cv::Mat gradientImageCopy = gradientImage.clone(); /* Only needed for visualization. @todo: delete */
    /* In the beginning, all pixel are not blocked. Matrix has two extra columns for the borders. */
    std::vector<std::vector<bool>> blockedPixels(gradientImage.rows, std::vector<bool>(gradientImage.cols + 2, false));

    /* Compute vertical seams and store them. */
    for (int i = 0; i < colsToRemove; i++) {
        std::vector<int> seam = seam::seamVertical(gradientImage, blockedPixels);
        if (seam.size() != 0)
            seamsVertical.emplace_back(seam);
        else {
            seamsVerticalBlockError(i);
            break;
        }

    }
    cv::imshow("vertical", gradientImage);

    /* Reset blocked pixels. In the beginning, all pixel are not blocked. Matrix has two extra rows
     * for the borders. */
    for (auto& row : blockedPixels) {
        std::fill(row.begin(), row.end(), 0);
        row.pop_back();
        row.pop_back();
    }
    blockedPixels.emplace_back(std::vector<bool>(gradientImage.cols, false));
    blockedPixels.emplace_back(std::vector<bool>(gradientImage.cols, false));

    /* Compute horizontal seams and store them. */
    for (int i = 0; i < rowsToRemove; i++) {
        std::vector<int> seam = seam::seamHorizontal(gradientImageCopy, blockedPixels);
        if (seam.size() != 0)
            seamsHorizontal.emplace_back(seam);
        else {
            seamsHorizontalBlockError(i);
            break;
        }
    }
    cv::imshow("horizontal", gradientImageCopy);

    gradientImage.release();
    gradientImageCopy.release();
}

void MainWindow::on_pbRemoveSeams_clicked()
{
    /* Check if seams were already computed. */
    if (seamsHorizontal.size() == 0 && seamsVertical.size() == 0) {
        noSeamsError();
        return;
    }

	std::sort(seamsVertical.begin(), seamsVertical.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
        return a[0] < b[0];
    });
    cv::Mat verticalDeletedImage;
    /* Remove all vertical seams that were computed earlier in ascending order. */
    seam::deleteSeamsVertical(originalImage, verticalDeletedImage, seamsVertical);

    seam::combineVerticalHorizontalSeams(seamsVertical, seamsHorizontal);

    std::sort(seamsHorizontal.begin(), seamsHorizontal.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
        return a[0] < b[0];
    });

    cv::Mat seamsDeletedImage;
    /* Remove all horizontal seams that were computed earlier in ascending order and adjusted for the already removed
       vertical seams. */
    seam::deleteSeamsHorizontal(verticalDeletedImage, seamsDeletedImage, seamsHorizontal);

    cv::imshow("Downscaled Image", seamsDeletedImage);
    modifiedImage = seamsDeletedImage;

    seamsHorizontal.clear();
    seamsVertical.clear();
    pbSaveImage->setEnabled(true);
}

void MainWindow::on_pbSaveImage_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Image"), "", tr("Images (*.png *.xpm *.jpg)"));
    if (fileName.isEmpty())
        disableGUI();
    else {
        QImage image = QtOpencvCore::img2qimg(modifiedImage);
        image.save(fileName);
    }
}

void MainWindow::setupUi()
{
    /* Boilerplate code */
    /*********************************************************************************************/
    resize(129, 211);
    QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    setSizePolicy(sizePolicy);
    setMinimumSize(QSize(129, 211));
    setMaximumSize(QSize(129, 211));
    centralWidget = new QWidget(this);
    centralWidget->setObjectName(QString("centralWidget"));
    
    horizontalLayout = new QHBoxLayout(centralWidget);
    verticalLayout = new QVBoxLayout();
    
    pbOpenImage = new QPushButton(QString("Open Image"), centralWidget);
    verticalLayout->addWidget(pbOpenImage);

    verticalLayout_3 = new QVBoxLayout();
    lCaption = new QLabel(QString("Remove"), centralWidget);
    lCaption->setEnabled(false);
    verticalLayout_3->addWidget(lCaption);
    
    horizontalLayout_3 = new QHBoxLayout();
    horizontalLayout_3->setObjectName(QString("horizontalLayout_3"));
    lCols = new QLabel(QString("Cols"), centralWidget);
    lCols->setEnabled(false);
    lRows = new QLabel(QString("Rows"), centralWidget);
    lRows->setEnabled(false);
    horizontalLayout_3->addWidget(lCols);
    horizontalLayout_3->addWidget(lRows);
    verticalLayout_3->addLayout(horizontalLayout_3);
    
    horizontalLayout_2 = new QHBoxLayout();
    sbCols = new QSpinBox(centralWidget);
    sbCols->setEnabled(false);
    horizontalLayout_2->addWidget(sbCols);
    sbRows = new QSpinBox(centralWidget);
    sbRows->setEnabled(false);
    horizontalLayout_2->addWidget(sbRows);
    verticalLayout_3->addLayout(horizontalLayout_2);
    verticalLayout->addLayout(verticalLayout_3);
    
    pbComputeSeams = new QPushButton(QString("Compute Seams"), centralWidget);
    pbComputeSeams->setEnabled(false);
    verticalLayout->addWidget(pbComputeSeams);
    
    pbRemoveSeams = new QPushButton(QString("Remove Seams"), centralWidget);
    pbRemoveSeams->setEnabled(false);
    verticalLayout->addWidget(pbRemoveSeams);

    pbSaveImage = new QPushButton(QString("Save Image"), centralWidget);
    pbSaveImage->setEnabled(false);
    verticalLayout->addWidget(pbSaveImage);

    
    verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
    verticalLayout->addItem(verticalSpacer);
    horizontalLayout->addLayout(verticalLayout);
    
    horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);
    horizontalLayout->addItem(horizontalSpacer);
    setCentralWidget(centralWidget);
    /*********************************************************************************************/
    
    
    /* Verbindung zwischen den Buttonklicks und den Methoden, die beim jeweiligen Buttonklick ausgefuehrt werden sollen */
    connect(pbOpenImage,    &QPushButton::clicked, this, &MainWindow::on_pbOpenImage_clicked);  
    connect(pbComputeSeams, &QPushButton::clicked, this, &MainWindow::on_pbComputeSeams_clicked); 
    connect(pbRemoveSeams,  &QPushButton::clicked, this, &MainWindow::on_pbRemoveSeams_clicked);
    connect(pbSaveImage, 	&QPushButton::clicked, this, &MainWindow::on_pbSaveImage_clicked);
}

void MainWindow::enableGUI()
{
    lCaption->setEnabled(true);
    
    lCols->setEnabled(true);
    lRows->setEnabled(true);
    
    sbCols->setEnabled(true);
    sbRows->setEnabled(true);
    
    pbComputeSeams->setEnabled(true);
    pbRemoveSeams->setEnabled(true);
    
    sbRows->setMinimum(0);
    sbRows->setMaximum(originalImage.rows);
    sbRows->setValue(2);
    
    sbCols->setMinimum(0);
    sbCols->setMaximum(originalImage.cols);
    sbCols->setValue(2);
}

void MainWindow::disableGUI()
{
    lCaption->setEnabled(false);
    
    lCols->setEnabled(false);
    lRows->setEnabled(false);
    
    sbCols->setEnabled(false);
    sbRows->setEnabled(false);
    
    pbComputeSeams->setEnabled(false);
    pbRemoveSeams->setEnabled(false);
    pbSaveImage->setEnabled(false);
}

void MainWindow::noSeamsError()
{
    QMessageBox messageBox;
    messageBox.critical(0, "No Seams", "No seams were calculated that can be removed.");
    messageBox.show();
}

void MainWindow::seamsVerticalBlockError(int i)
{
    QMessageBox messageBox;
    messageBox.critical(0, "Vertical Seams Blocked", QString("It is not possible to compute more non-blocking vertical "
                                                     "seams. %1 vertical seams were computed.").arg(i));
    messageBox.show();
}

void MainWindow::seamsHorizontalBlockError(int i)
{
    QMessageBox messageBox;
    messageBox.critical(0, "Horizontal Seams Blocked", QString("It is not possible to compute more non-blocking "
                                                       "horizontal seams. %1 horizontal seams were computed.").arg(i));
    messageBox.show();
}

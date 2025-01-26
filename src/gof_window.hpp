#pragma once

#include "cairomm/context.h"
#include "cairomm/refptr.h"
#include "gdkmm/general.h"
#include "gdkmm/pixbuf.h"
#include "glibmm/dispatcher.h"
#include "glibmm/refptr.h"
#include "gof.hpp"
#include "gtkmm/box.h"
#include "gtkmm/button.h"
#include "gtkmm/checkbutton.h"
#include "gtkmm/combobox.h"
#include "gtkmm/comboboxtext.h"
#include "gtkmm/drawingarea.h"
#include "gtkmm/entry.h"
#include "gtkmm/enums.h"
#include "gtkmm/frame.h"
#include "gtkmm/text.h"
#include "gtkmm/textbuffer.h"
#include "gtkmm/textview.h"
#include "gtkmm/window.h"
#include "sigc++/functors/mem_fun.h"

#include "gof_parallel.hpp"

#include <gtkmm.h>

struct runConfig_t {
    size_t iterations;
    size_t gpuThreads_x, gpuThreads_y;
    size_t cpuThreads;

    bool useGpu, useCpu;
    scheduleType schType;
    bool renderToBuff;
    bool recordSim;
};

struct runResult_t {
    double cpuTime; //in microseconds
    double gpuTime;
};

struct textEntry_t {
    Gtk::Box box;
    Gtk::Text text;
    Gtk::Entry entry;
};

enum openConfigType : uint8_t {
    CONFIG_TYPE_NONE,
    CONFIG_TYPE_INIT,
    CONFIG_TYPE_LOAD,
    CONFIG_TYPE_EXPORT,

    CONFIG_TYPE_MAX
}; 

class gofWindow_t : public Gtk::Window {
public:
    runConfig_t rConfig;
    runResult_t rResult;
    
    bMap_t map;
    mapRecord_t record;

    openConfigType currOpenConfig;

    Gtk::Box box, rightColumn, leftColumn;
    Gtk::Box initBox, cgpuBox;
    Gtk::Box loadOptions, exportOptions, initOptions;
    Gtk::Box gpuConfig, cpuConfig, baseConfig;
    Gtk::Frame gpuFrame, cpuFrame, baseConfigFrame;

    textEntry_t cpuThreadsEntry;
    Gtk::ComboBoxText cpuSchedule;

    textEntry_t gpuThreadsEntry_x, gpuThreadsEntry_y;

    textEntry_t loadPath;
    Gtk::Button loadAcceptBtn;

    textEntry_t exportPath;
    Gtk::Button exportAcceptBtn;

    textEntry_t initSize_x, initSize_y;
    Gtk::Button initAcceptBtn;

    Gtk::Button exportBtn, loadBtn, initBtn;
    Gtk::Button runBtn, cpuBtn, gpuBtn;
    Gtk::CheckButton renderCheckbox, recordCheckbox, exportRecording;

    textEntry_t iterationEntry;

    Gtk::TextView infoText;
    Gtk::DrawingArea dArea;
    Glib::Dispatcher dDispatcher;

    Glib::RefPtr<Gtk::TextBuffer> infoTextBuff;
    Glib::RefPtr<Gdk::Pixbuf> pixBuff;

    bool renderInBuff = false;

    gofWindow_t();
    void initConfigBoxes(void);
    void initLeftColumn(void);
    void initRightColumn(void);
    void init(void);

    void cpuBtnFunc(void);
    void gpuBtnFunc(void);
    void runBtnFunc(void);
    
    void closeConf(void);

    void openInitOptions(void);
    void openExportOptions(void);
    void openLoadOptions(void);

    void loadAcceptFunc(void);
    void exportAcceptFunc(void);
    void initAcceptFunc(void);

    bool updateRunConfig(void);
    void updatePixelBuff(void);
    void updateInfoTextBuff(void);
    void onDraw(const Cairo::RefPtr<Cairo::Context>& _cr, int _width, int _height);
    void onWorkerNotification(void);
    void toggleRender(void);
    void toggleRecord(void);
}; 

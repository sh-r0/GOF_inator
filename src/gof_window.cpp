#include <cstring>
#include <filesystem>
#include <iostream>
#include <format>
#include <memory>
#include <omp.h>
#include <chrono>
#include <memory>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <cctype>
#include <unordered_map>

#include "gof.hpp"
#include "gof_cuda.hpp"
#include "gof_io.hpp"
#include "gof_parallel.hpp"
#include "gof_window.hpp"
#include "gof.hpp"
#include "gtkmm/box.h"
#include "gtkmm/enums.h"
#include "sigc++/functors/mem_fun.h"

textEntry_t createEntry(const std::string& _txt, size_t _width){
    textEntry_t res = {Gtk::Box(), Gtk::Text(), Gtk::Entry() };
    res.box.set_orientation(Gtk::Orientation::HORIZONTAL);
    res.box.set_size_request(_width, -1);
    res.text.set_text(_txt);
    res.text.set_max_width_chars(8);
    res.text.set_editable(false);

    res.box.append(res.text);
    res.box.append(res.entry);

    return res;
}


gofWindow_t::gofWindow_t(){
    init();
}

void gofWindow_t::initLeftColumn(void){
    pixBuff = Gdk::Pixbuf::create(Gdk::Colorspace::RGB, false, 8, 256,256);
    dArea.set_size_request(256,256);
    dArea.set_draw_func(sigc::mem_fun(*this, &gofWindow_t::onDraw));
    dDispatcher.connect(sigc::mem_fun(*this, &gofWindow_t::onWorkerNotification));

    infoTextBuff = Gtk::TextBuffer::create();
    infoTextBuff->set_text("a\nb\nc");

    infoText.set_buffer(infoTextBuff);
    infoText.set_editable(false);
    infoText.set_cursor_visible(false);

    initBtn.set_label("Init");
    initBtn.signal_clicked().connect(sigc::mem_fun(*this, &gofWindow_t::openInitOptions));
    
    loadBtn.set_label("Load");
    loadBtn.signal_clicked().connect(sigc::mem_fun(*this, &gofWindow_t::openLoadOptions));
    
    exportBtn.set_label("Export");
    exportBtn.signal_clicked().connect(sigc::mem_fun(*this, &gofWindow_t::openExportOptions));
    
    initBtn.set_size_request(85, -1);
    loadBtn.set_size_request(85, -1);
    exportBtn.set_size_request(85, -1);

    initBox.set_orientation(Gtk::Orientation::HORIZONTAL);
    initBox.append(initBtn);
    initBox.append(loadBtn);
    initBox.append(exportBtn);

    leftColumn.set_orientation(Gtk::Orientation::VERTICAL);
    leftColumn.set_size_request(256,-1);

    leftColumn.append(dArea);
    leftColumn.append(infoText);
    leftColumn.append(initBox);

    return;
}

void gofWindow_t::initRightColumn(void) { 
    cpuBtn.set_label("CPU");
    cpuBtn.set_size_request(128,64);

    gpuBtn.set_label("GPU");
    gpuBtn.set_size_request(128,64);

    cpuBtn.signal_clicked().connect( sigc::mem_fun(*this, &gofWindow_t::cpuBtnFunc) );        
    gpuBtn.signal_clicked().connect( sigc::mem_fun(*this, &gofWindow_t::gpuBtnFunc) );        

    runBtn.set_label("Run");
    runBtn.set_size_request(256,64);
    runBtn.signal_clicked().connect(sigc::mem_fun(*this, &gofWindow_t::runBtnFunc));

    cgpuBox.set_orientation(Gtk::Orientation::HORIZONTAL);
    cgpuBox.set_size_request(256, 64);

    cgpuBox.append(gpuBtn);
    cgpuBox.append(cpuBtn);

    baseConfigFrame.set_label("Run config");
    baseConfig.set_orientation(Gtk::Orientation::VERTICAL);

    renderCheckbox.set_label("Render to image buffer");
    renderCheckbox.signal_toggled().connect(sigc::mem_fun(*this, &gofWindow_t::toggleRender));    
    recordCheckbox.set_label("Record simulation");
    recordCheckbox.signal_toggled().connect(sigc::mem_fun(*this, &gofWindow_t::toggleRecord));
    iterationEntry = createEntry("Iterations:", 256);

    baseConfig.append(iterationEntry.box);
    baseConfig.append(renderCheckbox);
    baseConfig.append(recordCheckbox);

    baseConfigFrame.set_child(baseConfig);

    rightColumn.set_orientation(Gtk::Orientation::VERTICAL);
    rightColumn.set_size_request(256, -1);

    rightColumn.append(runBtn);
    rightColumn.append(cgpuBox);
    rightColumn.append(baseConfigFrame);

    return;
}

void gofWindow_t::initConfigBoxes(void){
    //cpuConfig
    cpuConfig.set_orientation(Gtk::Orientation::VERTICAL);

    cpuThreadsEntry = createEntry("Threads: ", 256);
    cpuSchedule.set_entry_text_column(0);
    cpuSchedule.append("Dynamic");
    cpuSchedule.append("Static");
    cpuSchedule.append("Guided");
    cpuFrame.set_label("CPU config");

    cpuConfig.append(cpuThreadsEntry.box);
    cpuConfig.append(cpuSchedule);
    cpuFrame.set_child(cpuConfig);

    //gpuConfig
    gpuFrame.set_label("GPU config");
    gpuConfig.set_orientation(Gtk::Orientation::VERTICAL);
    
    gpuThreadsEntry_x = createEntry("X threads: ", 256);
    gpuThreadsEntry_y = createEntry("Y threads: ", 256);
    
    gpuConfig.append(gpuThreadsEntry_x.box);
    gpuConfig.append(gpuThreadsEntry_y.box);

    gpuFrame.set_child(gpuConfig);

    //initConfig
    initOptions.set_orientation(Gtk::Orientation::VERTICAL);
    initSize_x = createEntry("Map size x:", 192);
    initSize_y = createEntry("Map size y:", 192);
    
    initAcceptBtn.set_label("Accept");
    initAcceptBtn.signal_clicked().connect(sigc::mem_fun(*this, &gofWindow_t::initAcceptFunc));

    initOptions.append(initSize_x.box); 
    initOptions.append(initSize_y.box);
    initOptions.append(initAcceptBtn);

    //load
    loadOptions.set_orientation(Gtk::Orientation::VERTICAL);
    loadPath = createEntry("Path:", 192);
    loadAcceptBtn.set_label("Accept");
    loadAcceptBtn.signal_clicked().connect(sigc::mem_fun(*this, &gofWindow_t::loadAcceptFunc));

    loadOptions.append(loadPath.box);
    loadOptions.append(loadAcceptBtn);
    
    //export
    exportOptions.set_orientation(Gtk::Orientation::VERTICAL);
    exportPath = createEntry("Path:", 192);
    exportRecording.set_label("Export recording");

    exportAcceptBtn.set_label("Accept");
    exportAcceptBtn.signal_clicked().connect(sigc::mem_fun(*this, &gofWindow_t::exportAcceptFunc));
    
    exportOptions.append(exportPath.box);
    exportOptions.append(exportRecording);
    exportOptions.append(exportAcceptBtn); 

    return;
}

void gofWindow_t::init(void) {
    set_title("GOF");

    map = {};

    currOpenConfig = CONFIG_TYPE_NONE;

    initLeftColumn();
    initRightColumn();
    initConfigBoxes();

    box.set_orientation(Gtk::Orientation::HORIZONTAL);

    box.append(leftColumn);
    box.append(rightColumn);
    set_child(box);

    rConfig = {
        .iterations = 1,
        .useGpu = false,
        .useCpu = false,
        .renderToBuff = false,
        .recordSim = false
    };

    return;
}

bool getEntryVal(const textEntry_t& _entry, size_t& _val) {
    if(_entry.entry.get_text_length() == 0) return false;
    std::string txt = _entry.entry.get_text();
    for(char c : txt)
        if(!std::isdigit(c))
            return false;
    
    _val = std::stoi(txt);
    return true;
}

bool gofWindow_t::updateRunConfig(void){
    static std::unordered_map<std::string, scheduleType> mp;
    static bool initialized = false;
    if(!initialized) {
        mp.insert({"Dynamic", SCHEDULE_TYPE_DYNAMIC});
        mp.insert({"Static", SCHEDULE_TYPE_STATIC});
        mp.insert({"Guided", SCHEDULE_TYPE_GUIDED});
        initialized = true;
    }

    if(!rConfig.useCpu && !rConfig.useGpu)
        return false;

    if(!getEntryVal(iterationEntry, rConfig.iterations))
        return false;

    if(rConfig.useCpu){   
        if(!getEntryVal(cpuThreadsEntry, rConfig.cpuThreads))
            return false;
    
        std::string txt = cpuSchedule.get_active_text();
        if(mp.find(txt) == mp.end())
            return false;
        
        rConfig.schType = mp[txt];
    }

    if(rConfig.useGpu){ 
        if(!getEntryVal(gpuThreadsEntry_x, rConfig.gpuThreads_x))
            return false;
    
        if(!getEntryVal(gpuThreadsEntry_y, rConfig.gpuThreads_y))
            return false;
    }

    return true;
}

void gofWindow_t::updatePixelBuff() {
    const size_t buff_x = pixBuff->get_width(), buff_y = pixBuff->get_height();

    const size_t size_x = std::min(buff_x, map.x);
    const size_t size_y = std::min(buff_y, map.y);

    for(size_t x = 0; x < size_x; x++)
        for(size_t y = 0; y < size_y; y++){
            uint8_t val = map.at(x,y) ? 0 : 255;
            pixBuff->get_pixels()[x*3 + y*buff_x*3 + 0] = val;
            pixBuff->get_pixels()[x*3 + y*buff_x*3 + 1] = val;
            pixBuff->get_pixels()[x*3 + y*buff_x*3 + 2] = val;
        }
    dArea.queue_draw();
    return;
}

void gofWindow_t::updateInfoTextBuff() {
    static int32_t counter = 123;
    infoTextBuff->set_text(std::format("{}\nxd\nxd", counter++));

    return;
}

void gofWindow_t::onDraw(const Cairo::RefPtr<Cairo::Context>& _cr, int _width, int _height) {
    Gdk::Cairo::set_source_pixbuf(_cr, pixBuff);
    _cr->paint();
    return;
}

void gofWindow_t::onWorkerNotification(void) {
    queue_draw();

    return;
}

void gofWindow_t::toggleRender(void) {
    rConfig.renderToBuff = !rConfig.renderToBuff;

    return;
}

void gofWindow_t::toggleRecord(void) {
    rConfig.recordSim = !rConfig.recordSim;
    if(!rConfig.recordSim) {
        cleanMapRecord(record);
    } else {
        initMapRecord(record, map);
    }

    return;
}

void gofWindow_t::cpuBtnFunc(void){
    rConfig.useCpu = !rConfig.useCpu;
    cpuBtn.set_label(rConfig.useCpu ? "CPU: On" : "CPU: Off");
    if(rConfig.useCpu)
        rightColumn.append(cpuFrame);
    else
        rightColumn.remove(cpuFrame);

    return;
}

void gofWindow_t::gpuBtnFunc(void){
    rConfig.useGpu = !rConfig.useGpu;
    gpuBtn.set_label(rConfig.useGpu ? "GPU: On" : "GPU: Off"); 
    if(rConfig.useGpu)
        rightColumn.append(gpuFrame);
    else
        rightColumn.remove(gpuFrame);

    return;
}

void gofWindow_t::runBtnFunc(void){    
    if(!updateRunConfig()){ 
        infoTextBuff->set_text(std::format("Error parsing config information!"));
        return;
    }
   
    if(rConfig.recordSim) {
        if(map.x != record.x || map.y != record.y){
            cleanMapRecord(record);
            initMapRecord(record, map);
        }
    }

    std::string resText = std::format("Ran {} iterations\n", rConfig.iterations);

    if(rConfig.useCpu && rConfig.useGpu) {
        bMap_t tmp = cloneMap(map);

        if(rConfig.recordSim) {
            infoTextBuff->set_text("Can't run recording on this configuration!");
        } else {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now(), t2;

            for(size_t i = 0; i < rConfig.iterations; i++)
                parGof(map, rConfig.cpuThreads, rConfig.schType);

            t2 = std::chrono::steady_clock::now(); 
            rResult.cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(); 
            t1 = std::chrono::steady_clock::now(); 

            cudaGof(map, rConfig.gpuThreads_x, rConfig.gpuThreads_y, rConfig.iterations);

            t2 = std::chrono::steady_clock::now(); 
            rResult.gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(); 

            bool resultsSame = compareMap(map, tmp);

            resText += std::format("Cpu time: {}{}\n",
                    rResult.cpuTime > 1000 ? rResult.cpuTime / 1000 : rResult.cpuTime,
                    rResult.cpuTime > 1000 ? "seconds" : "milliseconds"
                    );


            resText += std::format("Gpu time: {}{}\n",
                    rResult.gpuTime > 1000 ? rResult.gpuTime / 1000 : rResult.gpuTime,
                    rResult.gpuTime > 1000 ? "seconds" : "milliseconds"
                    );

            resText += std::format("Got {} results", resultsSame ? "same" : "different");
        }

        goto _F_endRun;
    }

    if(rConfig.useCpu) {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now(), t2;

        if(!rConfig.renderToBuff && !rConfig.iterations) {
            for(size_t i = 0; i < rConfig.iterations; i++)
                parGof(map, rConfig.cpuThreads, rConfig.schType);
        } else {
            for(size_t i = 0; i < rConfig.iterations; i++) {
                parGof(map, rConfig.cpuThreads, rConfig.schType);
                if(rConfig.recordSim) pushMapState(record, map);
                /*if(rConfig.renderToBuff) {
                  updatePixelBuff();
                  }*/
            }
        }

        t2 = std::chrono::steady_clock::now(); 
        rResult.cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(); 

        resText += std::format("Cpu time: {}{}\n",
                rResult.cpuTime > 1000 ? rResult.cpuTime / 1000 : rResult.cpuTime,
                rResult.cpuTime > 1000 ? "seconds" : "milliseconds"
                );
    }                

    if(rConfig.useGpu) {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now(), t2;

        if(!rConfig.renderToBuff && !rConfig.iterations) {
            cudaGof(map, rConfig.gpuThreads_x, rConfig.gpuThreads_y, rConfig.iterations);
        } else {
            for(size_t i = 0; i < rConfig.iterations; i++) {
                cudaGof(map, rConfig.gpuThreads_x, rConfig.gpuThreads_y, 1);
                if(rConfig.recordSim) pushMapState(record, map);
                /*if(rConfig.renderToBuff) {
                  updatePixelBuff();
                  queue_draw();
                  } */
            }
        }

        t2 = std::chrono::steady_clock::now(); 
        rResult.gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count(); 

        resText += std::format("Gpu time: {}{}\n",
                rResult.gpuTime > 1000 ? rResult.gpuTime / 1000 : rResult.gpuTime,
                rResult.gpuTime > 1000 ? "seconds" : "milliseconds"
                );
    }

_F_endRun:
    updatePixelBuff();

    if(rConfig.recordSim)
        resText += std::format("Recording has {} frames", record.boardStates.size());

    infoTextBuff->set_text(resText);
    
    return;
}

void gofWindow_t::closeConf(void) {
    switch(currOpenConfig){
        case CONFIG_TYPE_INIT:
            leftColumn.remove(initOptions);
            break;
        case CONFIG_TYPE_LOAD:
            leftColumn.remove(loadOptions);
            break;
        case CONFIG_TYPE_EXPORT:
            leftColumn.remove(exportOptions);
            break;
    }
    return;
}

void gofWindow_t::openInitOptions(void) {
    closeConf(); 
    if(currOpenConfig != CONFIG_TYPE_INIT){
        leftColumn.append(initOptions);
        currOpenConfig = CONFIG_TYPE_INIT;
    } else currOpenConfig = CONFIG_TYPE_NONE;

    return;
}

void gofWindow_t::openLoadOptions(void) {
    closeConf(); 
    if(currOpenConfig != CONFIG_TYPE_LOAD){
        leftColumn.append(loadOptions);
        currOpenConfig = CONFIG_TYPE_LOAD;
    } else currOpenConfig = CONFIG_TYPE_NONE;
    
    return;
}

void gofWindow_t::openExportOptions(void) {
    closeConf(); 
    if(currOpenConfig != CONFIG_TYPE_EXPORT){
        leftColumn.append(exportOptions);
        currOpenConfig = CONFIG_TYPE_EXPORT;
    } else currOpenConfig = CONFIG_TYPE_NONE;
    
    return;
}

void gofWindow_t::initAcceptFunc(void) {
    size_t size_x, size_y;
    if(!getEntryVal(initSize_x, size_x)) return;
    if(!getEntryVal(initSize_y, size_y)) return;
    
    initMap(map, size_x, size_y);    
    randomizeMap(map);
    updatePixelBuff();

    return;
}

void gofWindow_t::exportAcceptFunc(void) {
    std::string txt = exportPath.entry.get_text();
    if(recordCheckbox.get_active()) {
        saveRecord(txt, record);    
    } else 
        saveMap(txt, map);

    return;
}

void gofWindow_t::loadAcceptFunc(void) {
    std::string txt = loadPath.entry.get_text();
    if(!std::filesystem::exists(txt)){
        infoTextBuff->set_text("Couldn't find specified file!");
        return;
    }
    loadMap(txt, map);
    updatePixelBuff();
    return;
}

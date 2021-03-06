function varargout = cervical_rangefinder(varargin)
% CERVICAL_RANGEFINDER MATLAB code for cervical_rangefinder.fig
%      CERVICAL_RANGEFINDER, by itself, creates a new CERVICAL_RANGEFINDER or raises the existing
%      singleton*.
%
%      H = CERVICAL_RANGEFINDER returns the handle to a new CERVICAL_RANGEFINDER or the handle to
%      the existing singleton*.
%
%      CERVICAL_RANGEFINDER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CERVICAL_RANGEFINDER.M with the given input arguments.
%
%      CERVICAL_RANGEFINDER('Property','Value',...) creates a new CERVICAL_RANGEFINDER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before cervical_rangefinder_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to cervical_rangefinder_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help cervical_rangefinder

% Last Modified by GUIDE v2.5 29-Mar-2017 16:36:14

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @cervical_rangefinder_OpeningFcn, ...
                   'gui_OutputFcn',  @cervical_rangefinder_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before cervical_rangefinder is made visible.
function cervical_rangefinder_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to cervical_rangefinder (see VARARGIN)

% Choose default command line output for cervical_rangefinder
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes cervical_rangefinder wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = cervical_rangefinder_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pb_select_database.
function pb_select_database_Callback(hObject, eventdata, handles)
% hObject    handle to pb_select_database (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[database_name, database_dir] = uigetfile();
database_full_path = [database_dir, database_name];

msg_update_status('database Selected:', handles);
msg_update_status(database_name, handles);
msg_update_status('database full path:', handles);
msg_update_status(database_full_path, handles);

% Load database 
% images data are stored in variable named img_data
load(database_full_path);

msg_update_status('-----------------------------',handles);
msg_update_status('database loaded!', handles);
msg_update_status('-----------------------------',handles);

% update GUI data
handles.database_name = database_name;
handles.database_dir = database_dir;
handles.database_full_path = database_full_path;
handles.img_data = images_temp;
handles.img_no = size(images_temp, 2);

% clear memory
clear('database_name', 'database_dir', 'database_full_path', 'images_temp');

% check label_mat available
handles.img_index = 1;
handles.label_data = [];
handles.h = [];
guidata(hObject, handles);
check_label_mat(hObject, eventdata, handles);


% Check existence of label mat 
function check_label_mat(hObject, eventdata, handles)
% hObject    handle to pb_select_database (see GCBO)
% handles    structure with handles and user data (see GUIDATA)

% Check file existence
[~,name,~] = fileparts(handles.database_name);
savename = [name, '_label.mat'];
if exist(savename, 'file') == 2
    
    % Construct a questdlg with three options
    choice = questdlg('Label Data Detected, continue label or start from beginning?', ...
	'Continue or Reset', ...
	'Continue From Last Time','Restart From Beginning', 'Continue From Last Time');

    % Handle response
    switch choice
        case 'Continue From Last Time'
            load(savename); % load label_data of last time
            handles.label_data = label_data;
            handles.img_index = size(label_data, 2) + 1;
            guidata(hObject, handles);
            
            msg_update_status('-----------------------------',handles);
            msg_update_status('Continue Labeling from Last Time ...', handles);
            msg_update_status('-----------------------------',handles);
            
            pb_start_Callback(hObject, eventdata, handles);

        case 'Restart From Beginning'
            msg_update_status('Restart From Beginning', handles);
            msg_update_status('-----------------------------',handles);
            pb_start_Callback(hObject, eventdata, handles);
    end
    
end
    


% Update real time message on the message box
function msg_update_status(msg, handles)
% msg : message to be displayed in message box

msgbox_str = get(handles.edit_msg_box,'String');
msgbox_str{end+1} =msg;
set(handles.edit_msg_box,'String',msgbox_str);


function edit_msg_box_Callback(hObject, eventdata, handles)
% hObject    handle to edit_msg_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_msg_box as text
%        str2double(get(hObject,'String')) returns contents of edit_msg_box as a double


% --- Executes during object creation, after setting all properties.
function edit_msg_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_msg_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pb_start.
function pb_start_Callback(hObject, eventdata, handles)
% hObject    handle to pb_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

load_image(handles);
create_label(hObject, eventdata,handles);


% --- Load next image to be labeld
function load_image(handles)
% handles   structure with handles and user data

axes(handles.axes_img);
imshow(handles.img_data{1, handles.img_index});
msg_update_status(['image ', num2str(handles.img_index), ' loaded'], handles);
msg_update_status('-----------------------------',handles);

% --- let user to draw polygon on figure
function create_label(hObject, eventdata,handles)
% handles   structure with handles and user data

handles.h = impoly(handles.axes_img);
guidata(hObject, handles);


% --- Executes on button press in pb_next.
function pb_next_Callback(hObject, eventdata, handles)
% hObject    handle to pb_next (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Save label data
BW = double(createMask(handles.h));
handles.label_data{handles.img_index} = BW;

msg_update_status(['image ', num2str(handles.img_index), ' label saved'], handles);
msg_update_status('-----------------------------',handles);

handles.img_index = handles.img_index + 1;
guidata(hObject, handles);

if handles.img_index > handles.img_no
    msg_update_status(['All images have been labeled, Press Save Button'], handles);
    msg_update_status('-----------------------------',handles);
else
    pb_start_Callback(hObject, eventdata, handles);
end


% --- Executes on button press in pb_save.
function pb_save_Callback(hObject, eventdata, handles)
% hObject    handle to pb_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Save label data
BW = double(createMask(handles.h));
handles.label_data{handles.img_index} = BW;

[~,name,~] = fileparts(handles.database_name);
savename = [name, '_label.mat'];

label_data = handles.label_data;
save(savename, 'label_data');
clear('label_data', 'savename');

msg_update_status(['Label Data Saved!'], handles);
msg_update_status('-----------------------------',handles);

%  Copyright [2014] [Jakub Konecny]
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
% 
%     http://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function [varargout] = myProcessOptions(options,varargin)
% Similar to processOptions, but case insensitive and
%   using a struct instead of a variable length list

options = toUpper(options);

for i = 1:2:length(varargin)
    if isfield(options,upper(varargin{i}))
        v = getfield(options,upper(varargin{i}));
        if isempty(v)
            varargout{(i+1)/2}=varargin{i+1};
        else
            varargout{(i+1)/2}=v;
        end
    else
        varargout{(i+1)/2}=varargin{i+1};
    end
end

end

function [o] = toUpper(o)
if ~isempty(o)
    fn = fieldnames(o);
    for i = 1:length(fn)
        o = setfield(o,upper(fn{i}),getfield(o,fn{i}));
    end
end
end